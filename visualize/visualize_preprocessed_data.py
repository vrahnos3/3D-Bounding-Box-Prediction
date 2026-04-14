import numpy as np
import open3d as o3d
import cv2
from pathlib import Path


def make_pcd(xyz, rgb=None, uniform_color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    if rgb is not None:
        colors = rgb.astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    elif uniform_color is not None:
        pcd.paint_uniform_color(uniform_color)

    return pcd


def load_raw_scene(scene_dir):
    scene_dir = Path(scene_dir)

    bbox3d = np.load(scene_dir / "bbox3d.npy").astype(np.float32)
    mask = np.load(scene_dir / "mask.npy").astype(bool)
    pc = np.load(scene_dir / "pc.npy").astype(np.float32)

    image = cv2.imread(str(scene_dir / "rgb.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return bbox3d, mask, pc, image


def extract_object_from_mask(scene_dir, object_id):
    _, mask, pc, image = load_raw_scene(scene_dir)

    obj_mask = mask[object_id]
    pc_hw3 = np.moveaxis(pc, 0, -1)

    xyz = pc_hw3[obj_mask]
    rgb = image[obj_mask]

    valid = np.isfinite(xyz).all(axis=1)
    valid &= ~(np.all(xyz == 0, axis=1))

    xyz = xyz[valid]
    rgb = rgb[valid]

    return xyz, rgb


def load_processed_object(processed_npz_path, object_id):
    data = np.load(processed_npz_path)

    model_input_points = data["model_input_points"][object_id]   # (K, C)
    centroid = data["centroids"][object_id]
    scale_factor = float(data["scale_factors"][object_id])

    xyz_norm = model_input_points[:, :3]

    rgb = None
    if model_input_points.shape[1] >= 6:
        rgb = model_input_points[:, 3:6]

    xyz_world = xyz_norm * scale_factor + centroid

    return xyz_world, xyz_norm, rgb, centroid, scale_factor


def add_coordinate_frame(size=0.2, origin=(0, 0, 0)):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


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
    colors = np.tile(np.array([[0., 0., 0.]]), (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_extracted_vs_downsampled(scene_dir, processed_npz_path, object_id, shift=0.0):
    """
    Raw extracted object vs processed/downsampled object in original coordinates.

    If shift == 0.0:
        overlay them
    If shift > 0:
        shift the downsampled version a little on x for visibility
    """
    data = np.load(processed_npz_path)
    bbox_world = data["bbox3d_world"][object_id]
    bbox_down_vis = bbox_world.copy()

    xyz_raw, _ = extract_object_from_mask(scene_dir, object_id)
    xyz_down, _, _, _, _ = load_processed_object(processed_npz_path, object_id)

    xyz_down_vis = xyz_down.copy()
    if shift > 0:
        xyz_down_vis[:, 0] += shift
        bbox_down_vis[:, 0] += shift

    pcd_raw = make_pcd(xyz_raw, uniform_color=[0.7, 0.7, 0.7])   # grey
    pcd_down = make_pcd(xyz_down_vis, uniform_color=[1.0, 0.0, 0.0])  # red

    bbox_raw_ls = bbox_lineset_from_corners(bbox_world)  # blue
    bbox_down_ls = bbox_lineset_from_corners(bbox_down_vis)  # magenta

    print(f"Raw extracted points: {len(xyz_raw)}")
    print(f"Processed/downsampled points: {len(xyz_down)}")

    o3d.visualization.draw_geometries(
        [pcd_raw, pcd_down, bbox_raw_ls, bbox_down_ls, add_coordinate_frame()],
        window_name=f"Object {object_id}: extracted (grey) vs downsampled (red)",
        width=1400,
        height=900
    )


def visualize_raw_normalized_vs_saved_normalized(scene_dir, processed_npz_path, object_id, gap=2.5):
    """
    Compare:
    - full extracted object normalized with the SAME centroid+scale
    - saved processed normalized object

    These are shown side by side in normalized coordinates.
    """
    xyz_raw, _ = extract_object_from_mask(scene_dir, object_id)

    data = np.load(processed_npz_path)
    centroid = data["centroids"][object_id]
    scale_factor = float(data["scale_factors"][object_id])
    bbox_norm = data["normalized_bbox3d"][object_id]

    _, xyz_norm_saved, _, _, _ = load_processed_object(processed_npz_path, object_id)

    xyz_norm_full = (xyz_raw - centroid) / scale_factor

    left = xyz_norm_full.copy()
    right = xyz_norm_saved.copy()

    bbox_left = bbox_norm.copy()
    bbox_right = bbox_norm.copy()

    left[:, 0] -= gap / 2.0
    right[:, 0] += gap / 2.0
    bbox_left[:, 0] -= gap / 2.0
    bbox_right[:, 0] += gap / 2.0

    pcd_left = make_pcd(left, uniform_color=[0.7, 0.7, 0.7])  # grey
    pcd_right = make_pcd(right, uniform_color=[0.0, 1.0, 0.0])  # green

    bbox_left_ls = bbox_lineset_from_corners(bbox_left)
    bbox_right_ls = bbox_lineset_from_corners(bbox_right)

    print(f"Full extracted normalized points: {len(xyz_norm_full)}")
    print(f"Saved processed normalized points: {len(xyz_norm_saved)}")

    o3d.visualization.draw_geometries(
        [pcd_left, pcd_right, bbox_left_ls, bbox_right_ls, add_coordinate_frame()],
        window_name=f"Object {object_id}: full normalized (grey) vs saved normalized (green)",
        width=1400,
        height=900
    )


if __name__ == "__main__":
    scene_dir = "C:/Users/panosvrachnos/Desktop/dl_challenge/scene_0001"
    processed_npz_path = "C:/Users/panosvrachnos/Desktop/dl_challenge/preprocess_data/scene_0001.npz"
    object_id = 1

    visualize_extracted_vs_downsampled(scene_dir, processed_npz_path, object_id, shift=0.3)
    visualize_raw_normalized_vs_saved_normalized(scene_dir, processed_npz_path, object_id, gap=2.5)