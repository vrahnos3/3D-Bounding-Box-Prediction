import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_processed_scene_files(data_path: str) -> List[Path]:
    """
    Returns all processed scene .npz files inside:
        data_path / preprocess_data
    """
    preprocess_dir = Path(data_path) / "preprocess_data"

    if not preprocess_dir.exists():
        raise FileNotFoundError(f"Preprocessed folder not found: {preprocess_dir}")

    scene_files = sorted(preprocess_dir.glob("*.npz"))

    if len(scene_files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {preprocess_dir}")

    return scene_files


def create_scene_splits(
    scene_files: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split scene files at the SCENE level, not object level.
    Returns dict with lists of filenames.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) != 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    scene_names = [f.name for f in scene_files]

    rng = random.Random(seed)
    rng.shuffle(scene_names)

    total_num_scenes = len(scene_names)
    n_train = int(total_num_scenes * train_ratio) # --> 180 scenes for train
    n_val = int(total_num_scenes * val_ratio) # --> 20 scenes for validation

    train_scenes = scene_names[:n_train]
    val_scenes = scene_names[n_train:n_train + n_val]
    test_scenes = scene_names[n_train + n_val:]

    return {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
    }


def get_or_create_splits(
    data_path: str,
    split_filename: str = "splits.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Loads splits.json if it exists.
    Otherwise creates it once and saves it in preprocess_data/.
    """
    preprocess_dir = Path(data_path) / "preprocess_data"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    split_file = preprocess_dir / split_filename

    if split_file.exists():
        with open(split_file, "r") as f:
            splits = json.load(f)
        return splits

    scene_files = get_processed_scene_files(data_path)
    splits = create_scene_splits(
        scene_files=scene_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


class InstanceDataset(Dataset):
    """
    One dataset item = one object instance.

    The split is created at the SCENE level,
    but the dataset returns OBJECT-level samples.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        split_filename: str = "splits.json",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        input_channels: int = 6,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unknown split: {split}")

        self.data_path = Path(data_path)
        self.preprocess_dir = self.data_path / "preprocess_data"
        self.split = split
        self.input_channels = input_channels
        if self.input_channels not in [3, 6]:
            raise ValueError(f"input_channels must be 3 or 6, got {self.input_channels}")

        splits = get_or_create_splits(
            data_path=data_path,
            split_filename=split_filename,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        split_scene_names = set(splits[split])

        # self.scene_files contains all files (scene) based on 'split'
        # So for 'split'=train it will contain all th *.npz which will be used for train
        self.scene_files = sorted([
            file for file in self.preprocess_dir.glob("*.npz")
            if file.name in split_scene_names
        ])

        if len(self.scene_files) == 0:
            raise ValueError(f"No scene files found for split='{split}'")

        # Each entry: (scene_file, object_idx)
        self.instance_index = self.build_instance_index()

    def build_instance_index(self) -> List[Tuple[Path, int]]:
        instance_index = []

        for scene_file in self.scene_files:
            data = np.load(scene_file)
            num_objects = len(data["object_ids"])

            for obj_idx in range(num_objects):
                instance_index.append((scene_file, obj_idx))

        return instance_index

    def __len__(self):
        return len(self.instance_index)

    def __getitem__(self, idx: int):
        scene_file, obj_idx = self.instance_index[idx]

        data = np.load(scene_file)
        points = data["model_input_points"][obj_idx].astype(np.float32)

        if points.shape[1] < self.input_channels:
            raise ValueError(
                f"Saved points have only {points.shape[1]} channels, "
                f"but input_channels={self.input_channels}"
            )

        points = points[:, :self.input_channels]

        sample = {
            "model_input_points": torch.from_numpy(
                points
            ),  # (K, 3) or (K, 6)

            "normalized_bbox3d": torch.from_numpy(
                data["normalized_bbox3d"][obj_idx].astype(np.float32)
            ),  # (8, 3)

            "bbox3d_world": torch.from_numpy(
                data["bbox3d_world"][obj_idx].astype(np.float32)
            ),  # (8, 3)

            "centroid": torch.from_numpy(
                data["centroids"][obj_idx].astype(np.float32)
            ),  # (3,)

            "scale_factor": torch.tensor(
                data["scale_factors"][obj_idx], dtype=torch.float32
            ),

            "num_real_points": torch.tensor(
                data["num_real_points"][obj_idx], dtype=torch.long
            ),

            "object_id": torch.tensor(
                data["object_ids"][obj_idx], dtype=torch.long
            ),

            "scene_name": scene_file.stem,
        }

        return sample


def create_dataloaders(
    data_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    split_filename: str = "splits.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    input_channels: int = 3,
):
    train_dataset = InstanceDataset(
        data_path=data_path,
        split="train",
        split_filename=split_filename,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        input_channels=input_channels,
    )

    val_dataset = InstanceDataset(
        data_path=data_path,
        split="val",
        split_filename=split_filename,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        input_channels=input_channels,
    )

    test_dataset = InstanceDataset(
        data_path=data_path,
        split="test",
        split_filename=split_filename,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        input_channels=input_channels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_path = "C:/Users/panosvrachnos/Desktop/dl_challenge"
    input_channels = 6

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=8,
        num_workers=4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        input_channels=input_channels
    )

    print("Train instances:", len(train_dataset))
    print("Val instances:", len(val_dataset))
    print("Test instances:", len(test_dataset))

    batch = next(iter(train_loader))
    print(batch["model_input_points"].shape)  # [B, K, 3] or [B, K, 6]
    print(batch["normalized_bbox3d"].shape)  # [B, 8, 3]
    print(batch["bbox3d_world"].shape)  # [B, 8, 3]

    print(batch["model_input_points"][0])
