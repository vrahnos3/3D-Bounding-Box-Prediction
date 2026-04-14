from tqdm import tqdm
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader


def validate_dataset(config: Dict):
    """
    Validate the consistency of the raw data
    """

    data_path = Path(config.get('data').get('data_path'))
    all_folder_paths = [f for f in data_path.iterdir() if f.is_dir()]
    print(f"Found {len(all_folder_paths)} folders to check.\n")
    corrupted_folders = []
    mismatched_folders = []

    for folder in tqdm(all_folder_paths, desc="Evaluating Data"):
        folder_name = folder.name
        bbox3d_path, mask_path, pc_path, rgb_path = get_scene_paths(folder_path=folder)

        if not (bbox3d_path.exists() and mask_path.exists() and pc_path.exists()):
            print(f"[{folder_name}] MISSING FILES")
            corrupted_folders.append(folder_name)
            continue

        try:
            bbox3d = np.load(bbox3d_path)
            mask = np.load(mask_path)
            pc = np.load(pc_path)

            N_bboxes, corners, coords = bbox3d.shape  # Expected: (N, 8, 3)
            N_masks, H_mask, W_mask = mask.shape  # Expected: (N, H, W)
            C_pc, H_pc, W_pc = pc.shape  # Expected: (3, H, W)

            # 3. Assert Alignments
            is_valid = True

            if N_bboxes != N_masks:
                print(f"[{folder_name}] MISMATCH: {N_bboxes} bboxes vs {N_masks} masks")
                is_valid = False

            if (H_mask != H_pc) or (W_mask != W_pc):
                print(f"[{folder_name}] RESOLUTION MISMATCH: Mask({H_mask}x{W_mask}) vs PC({H_pc}x{W_pc})")
                is_valid = False

            if not is_valid:
                mismatched_folders.append(folder_name)

        except Exception as e:
            print(f"[{folder_name}] CORRUPTED OR ERROR LOADING: {e}")
            corrupted_folders.append(folder_name)

    print("\n--- DATA VALIDATION SUMMARY ---")
    print(f"Total Folders Checked: {len(all_folder_paths)}")
    print(f"Perfectly Valid Folders: {len(all_folder_paths) - len(corrupted_folders) - len(mismatched_folders)}")
    print(f"Corrupted/Missing: {len(corrupted_folders)}")
    print(f"Mismatched Shapes: {len(mismatched_folders)}")


def get_scene_paths(folder_path: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Constructs and returns the standard file paths for a given scene folder.
    Returns: (bbox3d_path, mask_path, pc_path, rgb_path)
    """
    bbox3d_path = folder_path / "bbox3d.npy"
    mask_path = folder_path / "mask.npy"
    pc_path = folder_path / "pc.npy"
    rgb_path = folder_path / "rgb.jpg"

    return bbox3d_path, mask_path, pc_path, rgb_path


def scene_collate_fn(batch):
    # images = torch.stack([item["image"] for item in batch], dim=0)

    return {
        "image": [item["image"] for item in batch],
        "mask": [item["mask"] for item in batch],
        "bbox3d": [item["bbox3d"] for item in batch],
        "pc": [item["pc"] for item in batch],
    }


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def rename_subfolders(target_dir: Path, prefix: str = "scene"):

    # Get all subdirectories and sort them
    # .iterdir() gets everything; we filter using .is_dir()
    folders = sorted([f for f in target_dir.iterdir() if f.is_dir()])

    print(f"Found {len(folders)} folders. Renaming...")

    for index, folder_path in enumerate(folders, start=1):
        # Create the new name string
        new_name = f"{prefix}_{index:04d}"

        new_path = folder_path.with_name(new_name)

        try:
            folder_path.rename(new_path)
            print(f"Renamed: {folder_path.name} -> {new_name}")
        except FileExistsError:
            print(f"Error: Could not rename to {new_name} because it already exists.")
        except Exception as e:
            print(f"Error renaming {folder_path.name}: {e}")


# def make_dataloader(config: Dict, batch_size: int, shuffle: bool = False):
#     dataset = GlobalSceneDataset(config=config)
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         collate_fn=scene_collate_fn
#    )

if __name__ == "__main__":
    config = load_config('C:/Users/panosvrachnos/Desktop/Sereact_assignment/configs/config.yaml')
    data_path = Path(config.get('data').get('data_path'))

    validate_dataset(config=config)
    # rename_subfolders(target_dir=data_path)



