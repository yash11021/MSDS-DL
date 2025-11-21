from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# Notes

# Input:
#   lane boundary coordinates: track_left = left boundary coord = (B, 10, 2), track_right = right boundary coord = (B, 10, 2)
#   10 points, each with (x, z)
#   10 sequential points along the left/right boundary 

# Output:
#  target waypoints: (B, 3, 2) -> where the car should go
#  we need to return waypoints: (B, 3, 2) = 3 future points, each with (x, z)
#  tensor of predicted vehicle positions at the next `n_waypoints` time-steps

# Ego Coordinates
#   x: left(-) / right(+) relative to vehicle
#   y: up/down (height)
#   z: forward(+) / backward(-) relative to vehicle

# we don't care about y -> just bird's eye view (x, z)


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # input:
            # track_left = (B, n_track, 2)
            # track_right = (B, n_track, 2)
            # 2 sides, each with n_track points, each point with (x, z)
        input_dim = 2 * n_track * 2
        
        # 3 waypoints, each with (x, z)
        output_dim = n_waypoints * 2

        self.network = nn.Sequential(
            # (B, 40) -> (B, 256)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # (B, 256) -> (B, 256)
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # (B, 256) -> (B, 128)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # (B, 128) -> (B, 64)
            nn.Linear(128, 64),
            nn.ReLU(),
            # (B, 64) -> (B, 6)
            nn.Linear(64, output_dim),
        )


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]

        # flatten - fully connected layers expect 1D input
            # track_left = (B, n_track*2)
            # track_right = (B, n_track*2)
        left_flat = track_left.flatten(start_dim=1)
        right_flat = track_right.flatten(start_dim=1)

        # concatenate = (B, n_track*4)
        # at this point input_dim = 40
        x = torch.cat([left_flat, right_flat], dim=1) 

        x = self.network(x)
        
        # final linear layer will output (B, 6) 
        # reshape to (B, 3, 2)
        waypoints = x.view(B, self.n_waypoints, 2)  

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        num_heads = 4
        num_layers = 3

        # 1. learned query embeddings for each waypoint
        # (n_waypoints,) → (n_waypoints, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model) # (how many embeddings to store, size of each embedding vector)

        # 2. encode boundary points from (x, z) coords to d_model space
        # (B, 20, 2) -> (B, 20, d_model)
        self.boundary_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 3. cross-attention part
        # queries = waypoint embeddings = (B, 3, d_model)
        # keys/values = encoded boundaries (B, 20, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # 4. decode to (x, z) waypoints
        # (B, 3, d_model) -> (B, 3, 2)
        self.waypoint_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]

        # 1. concatenate left and right boundaries
        # (B, 10, 2) + (B, 10, 2) -> (B, 20, 2)
        boundaries = torch.cat([track_left, track_right], dim=1)

        # 2. encode boundary points to d_model space
        # (B, 20, 2) -> (B, 20, d_model)
        encoded_boundaries = self.boundary_encoder(boundaries)

        # 3. get query embeddings for waypoints
        # index [0, 1, 2] for the 3 waypoints
        # (n_waypoints,) -> (n_waypoints, d_model)
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        queries = self.query_embed(query_indices)

        # expand queries to batch dimension
        # (n_waypoints, d_model) -> (B, n_waypoints, d_model)
        # (3, 64) -> (B, 3, 64) - repeat for each item in batch
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        # 4. cross-attention: queries attend to encoded boundaries
        # queries = (B, 3, d_model)
        # encoded_boundaries = (B, 20, d_model) = keys/values
        # output = (B, 3, d_model)
        waypoint_features = self.transformer_decoder(
            tgt=queries,
            memory=encoded_boundaries,
        )

        # 5. decode to waypoint coordinates
        # (B, 3, d_model) -> (B, 3, 2)
        waypoints = self.waypoint_decoder(waypoint_features)

        return waypoints


class CNNPlanner(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.c3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn3 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.bn1(self.c1(x)))
            x = self.relu(self.bn2(self.c2(x)))
            x = self.relu(self.bn3(self.c3(x)))
            return x
        

    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # input: (B, 3, 96, 128)
        
        in_channels = 3
        output_dim = n_waypoints * 2  # 3 waypoints * 2 coords

        cnn_layers = [
            # layer 0: initial conv layer
            # input: (B, 3, 96, 128)
            # output: (B, 3, 48, 64)
            nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        ]

        c1 = in_channels
        n_blocks = 5

        # block 1: (B, 3, 48, 64) -> (B, 6, 24, 32)
        # block 2: (B, 6, 24, 32) -> (B, 12, 12, 16)
        # block 3: (B, 12, 12, 16) -> (B, 24, 6, 8)
        # block 4: (B, 24, 6, 8) -> (B, 48, 3, 4)
        # block 5: (B, 48, 3, 4) -> (B, 96, 1, 2)
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        # final conv layer to output_dim channels
        # (B, 96, 1, 2) -> (B, 6, 1, 2)
        cnn_layers.append(nn.Conv2d(c1, output_dim, kernel_size=1))

        # global average pooling
        # (B, 6, 1, 2) -> (B, 6, 1, 1)
        cnn_layers.append(nn.AdaptiveAvgPool2d(1))

        # flatten from (B, 6, 1, 1) to (B, 6)
        cnn_layers.append(nn.Flatten())

        self.network = nn.Sequential(*cnn_layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        B = image.shape[0]

        # normalize input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # pass through CNN network
        # output shape: (B, 6)
        x = self.network(x)

        # reshape to (B, n_waypoints, 2)
        waypoints = x.view(B, self.n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
