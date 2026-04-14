import os
import cv2
import numpy as np
import open3d as o3d
import torch


def load_data(data_dir):
    bbox3d = np.load(os.path.join(data_dir, "bbox3d.npy"))  # (N, 8, 3)
    mask = np.load(os.path.join(data_dir, "mask.npy"))      # (N, H, W)
    pc = np.load(os.path.join(data_dir, "pc.npy"))          # (3, H, W)
    rgb = cv2.imread((os.path.join(data_dir, "rgb.jpg")))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    print("------ Data Shape -------")
    print("bbox3d shape:", bbox3d.shape, "Dtype:", bbox3d.dtype)
    print("mask shape:", mask.shape, "Dtype:", mask.dtype)
    print("pc shape:", pc.shape, "Dtype:", pc.dtype)
    print("rgb shape:", rgb.shape, "Dtype:", rgb.dtype)

    return bbox3d, mask, pc, rgb


def extract_objects_from_masks(pc, rgb, mask, bbox3d=None, remove_invalid=True):
    """
    pc:    (3, H, W) float32
    rgb:   (H, W, 3) uint8
    mask:  (N, H, W) bool
    bbox3d:(N, 8, 3) optional
    """
    pc_hw3 = np.moveaxis(pc, 0, -1)  # (H, W, 3)

    objects = []
    for i in range(mask.shape[0]):
        m = mask[i]

        xyz = pc_hw3[m]   # (K, 3)
        colors = rgb[m]   # (K, 3)

        if remove_invalid:
            valid = np.isfinite(xyz).all(axis=1)
            valid &= ~(np.all(xyz == 0, axis=1))
            xyz = xyz[valid]
            colors = colors[valid]

        obj = {
            "object_id": i,
            "mask": m,
            "points_xyz": xyz,
            "colors_rgb": colors,
            "points_xyzrgb": np.hstack([xyz, colors.astype(np.float32)]),
        }

        if bbox3d is not None:
            obj["bbox3d"] = bbox3d[i]

        objects.append(obj)

    return objects


def object_to_open3d_pcd(obj):
    xyz = obj["points_xyz"]
    rgb = obj["colors_rgb"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    return pcd


def bbox_lineset_from_corners(corners):
    """
    corners: (8, 3)
    Assumes the 8 corners follow a box ordering compatible with these edges.
    """
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # red box
    colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_object_open3d(obj, show_bbox=True):
    if len(obj["points_xyz"]) == 0:
        print(f'Object {obj["object_id"]}: no valid points')
        return

    geoms = [object_to_open3d_pcd(obj)]

    if show_bbox and "bbox3d" in obj:
        geoms.append(bbox_lineset_from_corners(obj["bbox3d"]))

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f'Object {obj["object_id"]}',
        width=1200,
        height=900
    )


def visualize_all_objects_open3d(objects, max_points_per_object=20000, show_bbox=False):
    geoms = []

    for obj in objects:
        xyz = obj["points_xyz"]
        rgb = obj["colors_rgb"]

        if len(xyz) == 0:
            continue

        if len(xyz) > max_points_per_object:
            idx = np.random.choice(len(xyz), max_points_per_object, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        geoms.append(pcd)

        if show_bbox and "bbox3d" in obj:
            geoms.append(bbox_lineset_from_corners(obj["bbox3d"]))

    if not geoms:
        print("No valid geometry to display.")
        return

    o3d.visualization.draw_geometries(
        geoms,
        window_name="All objects",
        width=1400,
        height=1000
    )


def print_object_stats(objects):
    print("\n------ Extracted Objects -------")
    for obj in objects:
        print(
            f'Object {obj["object_id"]:2d}: '
            f'{obj["points_xyz"].shape[0]} points'
        )


def main():
    data_dir = "C:/Users/panosvrachnos/Desktop/dl_challenge/scene_0006"
    bbox3d, mask, pc, rgb = load_data(data_dir)
    objects = extract_objects_from_masks(pc, rgb, mask, bbox3d=bbox3d)

    print_object_stats(objects)

    # Visualize one object
    obj_id = 4
    visualize_object_open3d(objects[obj_id], show_bbox=True)

    # Visualize all objects together
    visualize_all_objects_open3d(objects, max_points_per_object=30000, show_bbox=True)


if __name__ == "__main__":
    main()