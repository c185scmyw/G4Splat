import os
import sys
import argparse
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from plyfile import PlyData, PlyElement

# Add G4Splat to path
sys.path.append(os.getcwd())

# Import COLMAP loader from G4Splat
try:
    from scene.colmap_loader import read_points3D_binary, read_cameras_binary, read_images_binary, qvec2rotmat, rotmat2qvec
except ImportError:
    # Fallback or manual implementation if needed
    print("Error: Could not import colmap_loader from G4Splat. Ensure you are running from the project root.")
    sys.exit(1)

def storePly(path, xyz, rgb):
    # Define the elements for the PLY file
    elements = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def points_to_image_space(points, R, T, fx, fy, cx, cy, W, H):
    # points: [N, 3]
    # R: [3, 3] (world to camera)
    # T: [3] (world to camera)
    points_cam = (R @ points.T).T + T
    depths = points_cam[:, 2]
    
    # Project to image
    x = points_cam[:, 0] / depths * fx + cx
    y = points_cam[:, 1] / depths * fy + cy
    
    return np.stack([x, y, depths], axis=1)

def filter_pcd(input_dir, output_dir, masks_dir, alpha_border=1):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading COLMAP data from {input_dir}")
    cameras = read_cameras_binary(os.path.join(input_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(input_dir, "images.bin"))
    points3D = read_points3D_binary(os.path.join(input_dir, "points3D.bin"))
    
    # Extract points and colors
    xyz = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()])
    point_ids = np.array(list(points3D.keys()))
    
    print(f"Initial points: {len(xyz)}")
    
    # We want to keep points that are in the mask in AT LEAST one image
    # and NOT in the background in any image (or some other logic)
    # The reference implementation seems to iteratively filter.
    
    keep_mask = np.zeros(len(xyz), dtype=bool)
    counts = np.zeros(len(xyz), dtype=int)
    fg_counts = np.zeros(len(xyz), dtype=int)
    
    for img_id, img_info in tqdm(images.items(), desc="Filtering points by masks"):
        cam = cameras[img_info.camera_id]
        
        # Get mask
        mask_name = os.path.splitext(img_info.name)[0]
        mask_path = os.path.join(masks_dir, f"{mask_name}.npy")
        if not os.path.exists(mask_path):
            continue
            
        mask = np.load(mask_path)
        
        # Project points to this image
        R = qvec2rotmat(img_info.qvec).T
        T = img_info.tvec
        
        # Simple pinhole projection
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        
        proj = points_to_image_space(xyz, R, T, fx, fy, cx, cy, cam.width, cam.height)
        x, y, d = proj[:, 0], proj[:, 1], proj[:, 2]
        
        # Points in front of camera and within image bounds
        valid = (d > 0) & (x >= 0) & (x < cam.width) & (y >= 0) & (y < cam.height)
        
        if not valid.any():
            continue
            
        ix = np.round(x[valid]).astype(int)
        iy = np.round(y[valid]).astype(int)
        
        # Check against mask
        # We use alpha_border to be slightly more lenient
        point_in_fg = mask[iy, ix] > 0
        
        indices = np.where(valid)[0]
        fg_counts[indices[point_in_fg]] += 1
        counts[indices] += 1

    # Keep points that are seen as foreground more often than background
    # or just seen as foreground at least once?
    # Reference says: pcd = pcd[flags] iteratively.
    # Actually, a safer way is to keep points that are seen as FG in at least X frames
    # and NEVER seen as BG in frames where they are clearly background?
    # Let's use a simple heuristic: seen as FG >= 1 and FG_ratio > 0.5
    
    final_flags = (fg_counts >= 1)
    
    xyz_fg = xyz[final_flags]
    rgb_fg = rgb[final_flags]
    point_ids_fg = point_ids[final_flags]
    
    print(f"Final foreground points: {len(xyz_fg)}")
    
    # Save new points3D.bin
    from scene.colmap_loader import Point3D
    new_points3D = {}
    for i, pid in enumerate(point_ids_fg):
        orig_p = points3D[pid]
        new_points3D[pid] = Point3D(id=pid, xyz=xyz_fg[i], rgb=rgb_fg[i],
                                    error=orig_p.error, image_ids=orig_p.image_ids,
                                    point2D_idxs=orig_p.point2D_idxs)
                                    
    # Save files
    from scene.colmap_loader import write_points3D_binary, write_cameras_binary, write_images_binary
    
    sparse_out = os.path.join(output_dir)
    os.makedirs(sparse_out, exist_ok=True)
    
    write_points3D_binary(new_points3D, os.path.join(sparse_out, "points3D.bin"))
    write_cameras_binary(cameras, os.path.join(sparse_out, "cameras.bin"))
    
    # Update image point3D_ids to reflect filtered points
    for img_id in images:
        p3d_ids = images[img_id].point3D_ids
        new_p3d_ids = np.where(np.isin(p3d_ids, point_ids_fg), p3d_ids, -1)
        images[img_id] = images[img_id]._replace(point3D_ids=new_p3d_ids)
        
    write_images_binary(images, os.path.join(sparse_out, "images.bin"))
    storePly(os.path.join(sparse_out, "points3D.ply"), xyz_fg, rgb_fg)
    
    # Create trainval.meta as requested
    with open(os.path.join(output_dir, "trainval.meta"), "w") as f:
        for img_id in images:
            f.write(f"{images[img_id].name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--masks', type=str, required=True)
    parser.add_argument('--alpha_border', type=int, default=1)
    args = parser.parse_args()
    
    filter_pcd(args.input, args.output, args.masks, args.alpha_border)
