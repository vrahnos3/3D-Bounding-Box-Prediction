import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.custom_model import ObjectPointNetRegressor
from train.trainer import Trainer
from data.data_loader import create_dataloaders
from data.data_utils import load_config
import argparse


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config.get("data").get("data_path"),
        batch_size=config.get("data").get("batch_size"),
        num_workers=config.get("data").get("num_workers"),
        train_ratio=config.get("data").get("train_ratio"),
        val_ratio=config.get("data").get("val_ratio"),
        test_ratio=config.get("data").get("test_ratio"),
        seed=config.get("data").get("seed"),
        input_channels=config.get("model").get("input_channels"),
    )

    print(f"Train instances: {len(train_dataset)}")
    print(f"Val instances: {len(val_dataset)}")
    print(f"Test instances: {len(test_dataset)}")

    model = ObjectPointNetRegressor(
        input_channels=config.get("model").get("input_channels"),
        dropout=config.get("model").get("dropout"),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.get("train").get("lr"),
        weight_decay=config.get("train").get("weight_decay"),
    )

    scheduler = config.get("train").get("scheduler")
    if scheduler is not None:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=config.get("train").get("scheduler"),
        use_amp=config.get("train").get("use_amp"),
        center_loss_weight=config.get("train").get("center_loss_weight"),
        experiment_name=config.get("experiment_name"),
    )

    best_val_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get("train").get("epochs"),
        checkpoint_dir=config.get("train").get("checkpoint_dir"),
    )

    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()