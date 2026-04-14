import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from data.data_utils import validate_dataset, load_config, get_scene_paths


def furthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling.

    Args:
        points: (N, C) array. First 3 columns must be XYZ.
        n_samples: number of points to keep.

    Returns:
        (n_samples, C) sampled points.
    """
    points = np.asarray(points)

    if len(points) <= n_samples:
        return points

    xyz = points[:, :3]

    points_left = np.arange(len(xyz))
    sample_inds = np.zeros(n_samples, dtype=int)
    dists = np.ones(len(xyz), dtype=np.float32) * np.inf

    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected)

    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        dist_to_last = ((xyz[last_added] - xyz[points_left]) ** 2).sum(axis=-1)
        dists[points_left] = np.minimum(dist_to_last, dists[points_left])

        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)

    return points[sample_inds]


def load_scene(scene_dir: Path):
    bbox3d = np.load(scene_dir / "bbox3d.npy").astype(np.float32)   # (M, 8, 3)
    mask = np.load(scene_dir / "mask.npy").astype(bool)             # (M, H, W)
    pc = np.load(scene_dir / "pc.npy").astype(np.float32)           # (3, H, W)

    image = cv2.imread(str(scene_dir / "rgb.jpg"))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {scene_dir / 'rgb.jpg'}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                   # (H, W, 3)

    return bbox3d, mask, pc, image


def extract_masked_points(pc: np.ndarray, image: np.ndarray, obj_mask: np.ndarray):
    pc_hw3 = np.moveaxis(pc, 0, -1)   # (H, W, 3)
    xyz = pc_hw3[obj_mask]            # (N, 3)
    rgb = image[obj_mask]             # (N, 3)
    return xyz, rgb


def remove_invalid_points(xyz: np.ndarray, rgb: np.ndarray):
    valid = np.isfinite(xyz).all(axis=1)
    valid &= ~(np.all(xyz == 0, axis=1))
    return xyz[valid], rgb[valid]


def build_point_features(xyz: np.ndarray, rgb: np.ndarray, use_rgb: bool = False):
    if use_rgb:
        rgb = rgb.astype(np.float32) / 255.0
        points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)  # (N, 6)
    else:
        points = xyz.astype(np.float32)                                  # (N, 3)
    return points


def normalize_pointcloud(points: np.ndarray, mode="unit_sphere"):
    """
    This function was found at: https://minibatchai.com/2021/08/07/FPS.html

    Normalize point cloud to zero mean and unit sphere.

    Args:
        points: (N, C), first 3 columns are XYZ.

    Returns:
        normalized_points: (N, C)
        centroid: (3,)
        scale_factor: float
    """
    points = points.copy()
    xyz = points[:, :3]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid

    if mode == "unit_sphere":
        scale_factor = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if scale_factor < 1e-8:
            scale_factor = 1.0
        xyz = xyz / scale_factor

    elif mode == "center_only":
        scale_factor = 1.0

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    points[:, :3] = xyz
    return points, centroid.astype(np.float32), float(scale_factor)


def normalize_bbox3d(bbox3d: np.ndarray, centroid: np.ndarray, scale_factor: float):
    bbox3d = bbox3d.astype(np.float32).copy()
    bbox3d = bbox3d - centroid
    bbox3d = bbox3d / scale_factor
    return bbox3d


def upsample_points(points: np.ndarray, target_num_points: int):
    num_points = len(points)
    extra = target_num_points - num_points
    extra_idx = np.random.choice(num_points, extra, replace=True)
    out = np.concatenate([points, points[extra_idx]], axis=0)

    perm = np.random.permutation(target_num_points)
    return out[perm]


def resize_points_to_input_dim(points: np.ndarray, target_input_dim: int):
    """
    Resize the Nx3 (or Nx6 if use_rgb values) to the desired input dimension.
    If N > target_input_dim, downsample with furthest point sampling method.
    If N < target_input_dim, upsample with replacement of the already exisiting points
    :return:
    """
    num_points = len(points)

    if num_points == 0:
        channels = points.shape[1] if points.ndim == 2 else 3
        return np.zeros((target_input_dim, channels), dtype=np.float32)

    if num_points > target_input_dim:
        return furthest_point_sampling(points, target_input_dim)

    if num_points < target_input_dim:
        return upsample_points(points, target_input_dim)

    return points


def process_object(
        pc: np.ndarray,
        image: np.ndarray,
        obj_mask: np.ndarray,
        bbox3d_world: np.ndarray,
        num_obj_points: int,
        use_rgb: bool,
        normalization_mode: str,
):
    xyz, rgb = extract_masked_points(pc, image, obj_mask)
    xyz, rgb = remove_invalid_points(xyz, rgb)

    num_real_points = int(xyz.shape[0])

    if num_real_points == 0:
        channels = 6 if use_rgb else 3
        model_input_points = np.zeros((num_obj_points, channels), dtype=np.float32)
        centroid = np.zeros(3, dtype=np.float32)
        scale_factor = 1.0
        normalized_bbox3d = bbox3d_world.astype(np.float32).copy()
    else:
        points = build_point_features(xyz, rgb, use_rgb=use_rgb)
        normalized_points, centroid, scale_factor = normalize_pointcloud(points, mode=normalization_mode)
        normalized_bbox3d = normalize_bbox3d(bbox3d_world, centroid, scale_factor)
        model_input_points = resize_points_to_input_dim(normalized_points, num_obj_points).astype(np.float32)

    return {
        "model_input_points": model_input_points,
        "normalized_bbox3d": normalized_bbox3d.astype(np.float32),
        "bbox3d_world": bbox3d_world.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "scale_factor": np.float32(scale_factor),
        "num_real_points": np.int32(num_real_points),
    }


def process_scene(scene_dir: Path, num_obj_points: int, use_rgb: bool, normalization_mode: str):
    bbox3d_all, mask_all, pc, image = load_scene(scene_dir)

    scene_objects = {
        "model_input_points": [],
        "normalized_bbox3d": [],
        "bbox3d_world": [],
        "centroids": [],
        "scale_factors": [],
        "num_real_points": [],
        "object_ids": [],
    }

    num_objects = int(bbox3d_all.shape[0])

    for obj_idx in range(num_objects):
        result = process_object(
            pc=pc,
            image=image,
            obj_mask=mask_all[obj_idx],
            bbox3d_world=bbox3d_all[obj_idx],
            num_obj_points=num_obj_points,
            use_rgb=use_rgb,
            normalization_mode=normalization_mode,
        )

        scene_objects["model_input_points"].append(result["model_input_points"])
        scene_objects["normalized_bbox3d"].append(result["normalized_bbox3d"])
        scene_objects["bbox3d_world"].append(result["bbox3d_world"])
        scene_objects["centroids"].append(result["centroid"])
        scene_objects["scale_factors"].append(result["scale_factor"])
        scene_objects["num_real_points"].append(result["num_real_points"])
        scene_objects["object_ids"].append(obj_idx)

    scene_objects["model_input_points"] = np.stack(scene_objects["model_input_points"], axis=0)
    scene_objects["normalized_bbox3d"] = np.stack(scene_objects["normalized_bbox3d"], axis=0)
    scene_objects["bbox3d_world"] = np.stack(scene_objects["bbox3d_world"], axis=0)
    scene_objects["centroids"] = np.stack(scene_objects["centroids"], axis=0)
    scene_objects["scale_factors"] = np.asarray(scene_objects["scale_factors"], dtype=np.float32)
    scene_objects["num_real_points"] = np.asarray(scene_objects["num_real_points"], dtype=np.int32)
    scene_objects["object_ids"] = np.asarray(scene_objects["object_ids"], dtype=np.int32)

    return scene_objects


def save_processed_scene(output_file: Path, scene_objects: dict):
    np.savez_compressed(
        output_file,
        model_input_points=scene_objects["model_input_points"],
        normalized_bbox3d=scene_objects["normalized_bbox3d"],
        bbox3d_world=scene_objects["bbox3d_world"],
        centroids=scene_objects["centroids"],
        scale_factors=scene_objects["scale_factors"],
        num_real_points=scene_objects["num_real_points"],
        object_ids=scene_objects["object_ids"],
    )


def is_valid_scene_dir(scene_dir: Path) -> bool:
    required_files = ["bbox3d.npy", "mask.npy", "pc.npy", "rgb.jpg"]
    return scene_dir.is_dir() and all((scene_dir / name).exists() for name in required_files)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess scene folders into object-level .npz files")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root containing scene subfolders")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output folder for preprocessed scene .npz files. Defaults to <data_path>/preprocess_data",
    )
    parser.add_argument("--input_dim", type=int, default=2048, help="Fixed number of points per object")
    parser.add_argument("--use_rgb", action="store_true", help="Store XYZRGB instead of XYZ only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed files")
    parser.add_argument("--normalization_mode", type=str, default="unit_sphere", choices=["center_only", "unit_sphere"])
    return parser.parse_args()


def main():
    args = parse_args()

    raw_root = Path(args.data_path)
    output_root = Path(args.output_path) if args.output_path else raw_root / "preprocess_data"
    output_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in raw_root.iterdir() if is_valid_scene_dir(p)])

    manifest = []
    skipped = 0

    for scene_dir in tqdm(scene_dirs, desc="Preprocessing scenes"):
        output_file = output_root / f"{scene_dir.name}.npz"

        if output_file.exists() and not args.overwrite:
            skipped += 1

            with np.load(output_file) as existing:
                num_objects = int(len(existing["object_ids"]))

            manifest.append(
                {
                    "raw_scene_name": scene_dir.name,
                    "raw_scene_path": str(scene_dir),
                    "processed_file": output_file.name,
                    "num_objects": num_objects,
                }
            )
            continue

        scene_objects = process_scene(
            scene_dir=scene_dir,
            num_obj_points=args.input_dim,
            use_rgb=args.use_rgb,
            normalization_mode=args.normalization_mode
        )
        save_processed_scene(output_file, scene_objects)

        manifest.append(
            {
                "raw_scene_name": scene_dir.name,
                "raw_scene_path": str(scene_dir),
                "processed_file": output_file.name,
                "num_objects": int(len(scene_objects["object_ids"])),
            }
        )

    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Processed files saved to: {output_root}")
    if skipped > 0 and not args.overwrite:
        print(f"Skipped {skipped} scenes because processed files already existed.")


if __name__ == "__main__":
    main()
