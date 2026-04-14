import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from data.data_loader import create_dataloaders
from data.data_utils import load_config
from model.custom_model import ObjectPointNetRegressor


def denormalize_bboxes(
    bbox_normalized: torch.Tensor,
    centroids: torch.Tensor,
    scale_factors: torch.Tensor,
) -> torch.Tensor:
    """
    bbox_normalized: (B, 8, 3)
    centroids:       (B, 3)
    scale_factors:   (B,)
    """
    return bbox_normalized * scale_factors.view(-1, 1, 1) + centroids.view(-1, 1, 3)


def box_center_from_corners(bbox: torch.Tensor) -> torch.Tensor:
    return bbox.mean(dim=1)


def compute_batch_metrics(
    pred_bbox_norm: torch.Tensor,
    gt_bbox_norm: torch.Tensor,
    pred_bbox_world: torch.Tensor,
    gt_bbox_world: torch.Tensor,
):
    """
    Returns per-sample metrics.
    """
    norm_corner_error = torch.linalg.norm(pred_bbox_norm - gt_bbox_norm, dim=-1).mean(dim=1)
    world_corner_error = torch.linalg.norm(pred_bbox_world - gt_bbox_world, dim=-1).mean(dim=1)

    pred_center_norm = box_center_from_corners(pred_bbox_norm)
    gt_center_norm = box_center_from_corners(gt_bbox_norm)
    norm_center_error = torch.linalg.norm(pred_center_norm - gt_center_norm, dim=-1)

    pred_center_world = box_center_from_corners(pred_bbox_world)
    gt_center_world = box_center_from_corners(gt_bbox_world)
    world_center_error = torch.linalg.norm(pred_center_world - gt_center_world, dim=-1)

    return {
        "norm_corner_error": norm_corner_error,
        "world_corner_error": world_corner_error,
        "norm_center_error": norm_center_error,
        "world_center_error": world_center_error,
    }


def load_scene_pointcloud(scene_dir: Path):
    pc = np.load(scene_dir / "pc.npy").astype(np.float32)          # (3, H, W)

    image = cv2.imread(str(scene_dir / "rgb.jpg"))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {scene_dir / 'rgb.jpg'}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pc_hw3 = np.moveaxis(pc, 0, -1)                               # (H, W, 3)
    xyz = pc_hw3.reshape(-1, 3)
    rgb = image.reshape(-1, 3)

    valid = np.isfinite(xyz).all(axis=1)
    valid &= ~(np.all(xyz == 0, axis=1))

    xyz = xyz[valid]
    rgb = rgb[valid]

    return xyz, rgb


def make_pcd(xyz: np.ndarray, rgb: np.ndarray | None = None, uniform_color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    elif uniform_color is not None:
        pcd.paint_uniform_color(uniform_color)

    return pcd


def bbox_lineset_from_corners(corners: np.ndarray, color=(1.0, 0.0, 0.0)):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = np.tile(np.array(color, dtype=np.float64)[None, :], (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_scene_prediction(
    data_path: str,
    scene_name: str,
    pred_bbox_world: np.ndarray,
    gt_bbox_world: np.ndarray,
    window_name: str | None = None,
):
    scene_dir = Path(data_path) / scene_name
    xyz, rgb = load_scene_pointcloud(scene_dir)

    scene_pcd = make_pcd(xyz, rgb=rgb)
    pred_box = bbox_lineset_from_corners(pred_bbox_world, color=(1.0, 0.0, 0.0))   # red
    gt_box = bbox_lineset_from_corners(gt_bbox_world, color=(0.0, 0.0, 1.0))       # blue
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    if window_name is None:
        window_name = f"{scene_name} | GT=blue | Pred=red"

    o3d.visualization.draw_geometries(
        [scene_pcd, gt_box, pred_box, frame],
        window_name=window_name,
        width=1400,
        height=900,
    )


def visualize_object_prediction(
    points: np.ndarray,
    pred_bbox_norm: np.ndarray,
    gt_bbox_norm: np.ndarray,
    window_name: str = "Object crop | GT=blue | Pred=red",
):
    """
    Visualize in the normalized/local frame.
    """
    xyz = points[:, :3]
    rgb = points[:, 3:6] if points.shape[1] >= 6 else None

    pcd = make_pcd(xyz, rgb=rgb)
    pred_box = bbox_lineset_from_corners(pred_bbox_norm, color=(1.0, 0.0, 0.0))
    gt_box = bbox_lineset_from_corners(gt_bbox_norm, color=(0.0, 0.0, 1.0))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    o3d.visualization.draw_geometries(
        [pcd, gt_box, pred_box, frame],
        window_name=window_name,
        width=1200,
        height=800,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Test / inference script for 3D bbox regression")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to best checkpoint")
    parser.add_argument("--num_visualizations", type=int, default=5, help="How many test samples to visualize")
    parser.add_argument("--skip_visualization", action="store_true", help="Disable Open3D visualization")
    parser.add_argument("--save_results_json", type=str, default=None, help="Optional path to save final metrics")
    parser.add_argument("--scene_name", type=str, default=None, help="Visualize only this scene")
    parser.add_argument("--list_test_scenes", action="store_true",
                        help="List available scenes in the test split and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_dataset, _, _, test_loader = create_dataloaders(
        data_path=config.get("data").get("data_path"),
        batch_size=config.get("data").get("batch_size"),
        num_workers=config.get("data").get("num_workers"),
        train_ratio=config.get("data").get("train_ratio"),
        val_ratio=config.get("data").get("val_ratio"),
        test_ratio=config.get("data").get("test_ratio"),
        seed=config.get("data").get("seed"),
        input_channels=config.get("model").get("input_channels"),
    )
    available_test_scenes = []

    for scene_file in test_dataset.scene_files:
        data = np.load(scene_file)
        num_objects = len(data["object_ids"])
        available_test_scenes.append((scene_file.stem, num_objects))

    available_test_scene_names = {scene_name for scene_name, _ in available_test_scenes}

    if args.list_test_scenes:
        print("Available test scenes:")
        for scene_name, num_objects in available_test_scenes:
            print(f"{scene_name} | {num_objects} objects")
        return

    if args.scene_name is not None and args.scene_name not in available_test_scene_names:
        available_str = ", ".join(scene_name for scene_name, _ in available_test_scenes)
        raise ValueError(
            f"Scene '{args.scene_name}' is not in the test split. "
            f"Available scenes: {available_str}"
        )
    print(f"Test instances: {len(test_dataset)}")

    model = ObjectPointNetRegressor(
        input_channels=config.get("model").get("input_channels"),
        dropout=config.get("model").get("dropout"),
    ).to(device)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = str(Path(config.get("train").get("checkpoint_dir")) / "best.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    total_samples = 0
    sum_norm_corner_error = 0.0
    sum_world_corner_error = 0.0
    sum_norm_center_error = 0.0
    sum_world_center_error = 0.0

    visualization_cache = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            points = batch["model_input_points"].to(device)            # (B, K, C)
            gt_bbox_norm = batch["normalized_bbox3d"].to(device)       # (B, 8, 3)
            gt_bbox_world = batch["bbox3d_world"].to(device)           # (B, 8, 3)
            centroids = batch["centroid"].to(device)                   # (B, 3)
            scale_factors = batch["scale_factor"].to(device)           # (B,)

            pred_bbox_norm = model(points)
            pred_bbox_world = denormalize_bboxes(pred_bbox_norm, centroids, scale_factors)

            metrics = compute_batch_metrics(
                pred_bbox_norm=pred_bbox_norm,
                gt_bbox_norm=gt_bbox_norm,
                pred_bbox_world=pred_bbox_world,
                gt_bbox_world=gt_bbox_world,
            )

            batch_size = points.shape[0]
            total_samples += batch_size

            sum_norm_corner_error += metrics["norm_corner_error"].sum().item()
            sum_world_corner_error += metrics["world_corner_error"].sum().item()
            sum_norm_center_error += metrics["norm_center_error"].sum().item()
            sum_world_center_error += metrics["world_center_error"].sum().item()

            if not args.skip_visualization and len(visualization_cache) < args.num_visualizations:

                scene_names = batch["scene_name"]
                object_ids = batch["object_id"].cpu().numpy()

                for i in range(batch_size):
                    if args.scene_name is not None and scene_names[i] != args.scene_name:
                        continue

                    if len(visualization_cache) >= args.num_visualizations:
                        break

                    visualization_cache.append({
                        "scene_name": scene_names[i],
                        "object_id": int(object_ids[i]),
                        "points": points[i].cpu().numpy(),
                        "pred_bbox_norm": pred_bbox_norm[i].cpu().numpy(),
                        "gt_bbox_norm": gt_bbox_norm[i].cpu().numpy(),
                        "pred_bbox_world": pred_bbox_world[i].cpu().numpy(),
                        "gt_bbox_world": gt_bbox_world[i].cpu().numpy(),
                    })

    results = {
        "checkpoint_path": checkpoint_path,
        "num_test_instances": total_samples,
        "normalized_corner_error": sum_norm_corner_error / total_samples,
        "world_corner_error": sum_world_corner_error / total_samples,
        "normalized_center_error": sum_norm_center_error / total_samples,
        "world_center_error": sum_world_center_error / total_samples,
    }

    print("\nFinal test results")
    print(json.dumps(results, indent=2))

    if args.save_results_json is not None:
        save_path = Path(args.save_results_json)
    else:
        experiment_name = config.get("experiment_name", "experiment")
        save_path = Path(config.get("train").get("checkpoint_dir")) / f"test_results_{experiment_name}.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved test results to: {save_path}")

    if not args.skip_visualization:
        print(f"\nShowing {len(visualization_cache)} visualization examples...")

        for idx, item in enumerate(visualization_cache, start=1):
            scene_name = item["scene_name"]
            object_id = item["object_id"]

            visualize_object_prediction(
                points=item["points"],
                pred_bbox_norm=item["pred_bbox_norm"],
                gt_bbox_norm=item["gt_bbox_norm"],
                window_name=f"[{idx}] Object crop | {scene_name} | obj={object_id} | GT=blue | Pred=red",
            )

            visualize_scene_prediction(
                data_path=config.get("data").get("data_path"),
                scene_name=scene_name,
                pred_bbox_world=item["pred_bbox_world"],
                gt_bbox_world=item["gt_bbox_world"],
                window_name=f"[{idx}] Whole scene | {scene_name} | obj={object_id} | GT=blue | Pred=red",
            )


if __name__ == "__main__":
    main()