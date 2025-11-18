"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner --num_epoch 50 --lr 1e-3
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    loss_func = torch.nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # Initialize metrics
        train_metric = PlannerMetric()
        train_metric.reset()

        train_losses = []

        model.train()

        for batch in train_data:
            # track_left: (B, 10, 2) - left lane boundary points
            # track_right: (B, 10, 2) - right lane boundary points
            # waypoints: (B, 3, 2) - ground truth waypoints 
            # waypoints_mask: (B, 3) - boolean mask for valid waypoints
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            optimizer.zero_grad()

            # forward
            # (B, 3, 2)
            pred_waypoints = model(track_left, track_right)

            # (B, 3, 2)
            losses = loss_func(pred_waypoints, waypoints)

            # waypoints_mask: (B, 3) -> (B, 3, 1) to broadcast over coordinates
            loss_masked = losses * waypoints_mask.unsqueeze(-1)

            loss = loss_masked.sum() / waypoints_mask.sum()

            loss.backward()
            optimizer.step()

            train_metric.add(pred_waypoints, waypoints, waypoints_mask)
            train_losses.append(loss.item())

            global_step += 1

        train_results = train_metric.compute()
        train_loss = np.mean(train_losses)

        val_metric = PlannerMetric()
        val_metric.reset()
        val_losses = []

        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred_waypoints = model(track_left, track_right)

                losses = loss_func(pred_waypoints, waypoints)
                loss_masked = losses * waypoints_mask.unsqueeze(-1)
                loss = loss_masked.sum() / waypoints_mask.sum()

                val_metric.add(pred_waypoints, waypoints, waypoints_mask)
                val_losses.append(loss.item())

        val_results = val_metric.compute()
        val_loss = np.mean(val_losses)

        logger.add_scalar("train/loss", train_loss, epoch)
        logger.add_scalar("train/longitudinal_error", train_results["longitudinal_error"], epoch)
        logger.add_scalar("train/lateral_error", train_results["lateral_error"], epoch)
        logger.add_scalar("train/l1_error", train_results["l1_error"], epoch)

        logger.add_scalar("val/loss", val_loss, epoch)
        logger.add_scalar("val/longitudinal_error", val_results["longitudinal_error"], epoch)
        logger.add_scalar("val/lateral_error", val_results["lateral_error"], epoch)
        logger.add_scalar("val/l1_error", val_results["l1_error"], epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"train_long={train_results['longitudinal_error']:.4f} "
                f"train_lat={train_results['lateral_error']:.4f} "
                f"val_long={val_results['longitudinal_error']:.4f} "
                f"val_lat={val_results['lateral_error']:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    print(f"\nFinal Results:")
    print(f"  Longitudinal error: {val_results['longitudinal_error']:.4f} (target: < 0.2)")
    print(f"  Lateral error: {val_results['lateral_error']:.4f} (target: < 0.6)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparameters
    # parser.add_argument("--hidden_dim", type=int, default=128)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
