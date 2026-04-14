from pathlib import Path
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = False,
        center_loss_weight: float = 0.5,
        experiment_name: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.center_loss_weight = center_loss_weight

        self.corner_loss_fn = nn.SmoothL1Loss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.experiment_name = experiment_name
        self.log_dir = f'./runs/{self.experiment_name}'
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None

    @staticmethod
    def box_center_from_corners(bbox: torch.Tensor) -> torch.Tensor:
        """
        bbox: (B, 8, 3)
        returns: (B, 3)
        """
        return bbox.mean(dim=1)

    def compute_loss(self, pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> Dict[str, torch.Tensor]:
        corner_loss = self.corner_loss_fn(pred_bbox, gt_bbox)

        pred_center = self.box_center_from_corners(pred_bbox)
        gt_center = self.box_center_from_corners(gt_bbox)
        center_loss = self.corner_loss_fn(pred_center, gt_center)

        total_loss = corner_loss + self.center_loss_weight * center_loss

        return {
            "loss": total_loss,
            "corner_loss": corner_loss,
            "center_loss": center_loss,
        }

    @staticmethod
    def compute_metrics(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> Dict[str, float]:
        """
        Metrics in the same frame as the targets.
        """
        corner_error = torch.linalg.norm(pred_bbox - gt_bbox, dim=-1).mean().item()

        pred_center = pred_bbox.mean(dim=1)
        gt_center = gt_bbox.mean(dim=1)
        center_error = torch.linalg.norm(pred_center - gt_center, dim=-1).mean().item()

        return {
            "corner_error": corner_error,
            "center_error": center_error,
        }

    def run_one_epoch(self, dataloader, training: bool = True) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_corner_loss = 0.0
        total_center_loss = 0.0
        total_corner_error = 0.0
        total_center_error = 0.0
        num_batches = 0

        for batch in dataloader:
            points = batch["model_input_points"].to(self.device)      # (B, K, C)
            gt_bbox = batch["normalized_bbox3d"].to(self.device)      # (B, 8, 3)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred_bbox = self.model(points)
                    loss_dict = self.compute_loss(pred_bbox, gt_bbox)
                    loss = loss_dict["loss"]

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            metrics = self.compute_metrics(pred_bbox.detach(), gt_bbox.detach())

            total_loss += loss_dict["loss"].item()
            total_corner_loss += loss_dict["corner_loss"].item()
            total_center_loss += loss_dict["center_loss"].item()
            total_corner_error += metrics["corner_error"]
            total_center_error += metrics["center_error"]
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Dataloader produced zero batches.")

        return {
            "loss": total_loss / num_batches,
            "corner_loss": total_corner_loss / num_batches,
            "center_loss": total_center_loss / num_batches,
            "corner_error": total_corner_error / num_batches,
            "center_error": total_center_error / num_batches,
        }

    def train(self, train_loader, val_loader, epochs: int, checkpoint_dir: str):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        pbar = tqdm(range(1, epochs + 1), desc="Epochs", dynamic_ncols=True)
        log_file = checkpoint_dir / f"train_log_{self.experiment_name}.txt"

        for epoch in pbar:
            train_stats = self.run_one_epoch(train_loader, training=True)
            val_stats = self.run_one_epoch(val_loader, training=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "lr": f"{current_lr:.6f}",
                "train_loss": f"{train_stats['loss']:.4f}",
                "val_loss": f"{val_stats['loss']:.4f}",
                "val_corner_err": f"{val_stats['corner_error']:.4f}",
            })
            log_line = (
                f"Epoch {epoch:03d} | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_stats['loss']:.4f} | "
                f"val_loss={val_stats['loss']:.4f} | "
                f"val_corner_err={val_stats['corner_error']:.4f}"
            )
            tqdm.write(log_line)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

            # print(
            #     f"Epoch {epoch:03d} | "
            #     f"lr={current_lr:.6f} | "
            #     f"train_loss={train_stats['loss']:.4f} | "
            #     f"val_loss={val_stats['loss']:.4f} | "
            #     f"val_corner_err={val_stats['corner_error']:.4f} | "
            #     f"val_center_err={val_stats['center_error']:.4f}"
            # )

            last_ckpt = checkpoint_dir / "last.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_stats["loss"],
                },
                last_ckpt,
            )

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_ckpt = checkpoint_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_stats["loss"],
                    },
                    best_ckpt,
                )
            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_stats["loss"], epoch)
                self.writer.add_scalar("val/loss", val_stats["loss"], epoch)
                self.writer.add_scalar("train/corner_error", train_stats["corner_error"], epoch)
                self.writer.add_scalar("val/corner_error", val_stats["corner_error"], epoch)
                self.writer.add_scalar("lr", current_lr, epoch)
        if self.writer is not None:
            self.writer.close()

        return best_val_loss
