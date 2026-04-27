import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

# Add G4Splat to path
sys.path.append(os.getcwd())

# Import COLMAP loader from G4Splat
try:
    from scene.colmap_loader import read_points3D_binary, read_cameras_binary, read_images_binary, qvec2rotmat, rotmat2qvec, write_points3D_binary, write_cameras_binary, write_images_binary
except ImportError:
    print("Error: Could not import colmap_loader from G4Splat. Ensure you are running from the project root.")
    sys.exit(1)

def HomoRotX(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def storePly(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    elements = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def calibrate_scale(input_dir, output_dir, arkit_dir, min_scale=0.8, max_scale=5.0):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading COLMAP data from {input_dir}")
    cameras = read_cameras_binary(os.path.join(input_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(input_dir, "images.bin"))
    points3D = read_points3D_binary(os.path.join(input_dir, "points3D.bin"))
    
    # Filename matching for ARKit
    arkit_cam_trans = []
    colmap_cam_trans = []
    
    # Sort images by name to be consistent if needed, but not strictly required for matching
    image_list = sorted(images.values(), key=lambda x: x.name)
    
    matched_count = 0
    for img_info in image_list:
        frame_id = os.path.splitext(img_info.name)[0]
        arkit_path = os.path.join(arkit_dir, f"{frame_id}.json")
        
        if not os.path.exists(arkit_path):
            continue
            
        with open(arkit_path, 'r') as f:
            arkit_data = json.load(f)
            
        # ARKit world to camera transformation
        cameraPoseARFrame = np.array(arkit_data['cameraPoseARFrame']).reshape([4, 4])
        # Convert ARKit coordinate system to COLMAP style (optional but standard in 3DRealCar)
        cameraPoseARFrame = HomoRotX(np.pi) @ cameraPoseARFrame
        
        # COLMAP camera to world transformation
        qvec = img_info.qvec[[1, 2, 3, 0]].copy() # x,y,z,w -> w,x,y,z for R.from_quat? 
        # Actually G4Splat qvec is [w, x, y, z]. Scipy R expects [x, y, z, w].
        qvec_scipy = img_info.qvec[[1, 2, 3, 0]]
        tvec = img_info.tvec
        
        Rt = np.eye(4)
        Rt[:3, :3] = R.from_quat(qvec_scipy).as_matrix()
        Rt[:3, 3] = tvec
        
        C2W = np.linalg.inv(Rt)
        
        arkit_cam_trans.append(cameraPoseARFrame[:3, 3])
        colmap_cam_trans.append(C2W[:3, 3])
        matched_count += 1

    print(f"Matched {matched_count}/{len(images)} frames with ARKit data")
    
    if matched_count < 2:
        print("Error: Not enough matched frames to calibrate scale.")
        sys.exit(1)
        
    arkit_cam_trans = np.array(arkit_cam_trans)
    colmap_cam_trans = np.array(colmap_cam_trans)
    
    # PCA alignment to find scale
    colmap_pca = PCA().fit(colmap_cam_trans)
    arkit_pca = PCA().fit(arkit_cam_trans)
    
    # Project to PCA space to compare ranges
    colmap_proj = colmap_pca.transform(colmap_cam_trans)
    arkit_proj = arkit_pca.transform(arkit_cam_trans)
    
    # Calculate scale as ratio of spans
    colmap_span = colmap_proj.max(0) - colmap_proj.min(0)
    arkit_span = arkit_proj.max(0) - arkit_proj.min(0)
    
    # Avoid division by zero
    colmap_span[colmap_span < 1e-6] = 1e-6
    scales = arkit_span / colmap_span
    
    # Use mean of X and Y scale (most stable)
    final_scale = scales[:2].mean()
    
    print(f"Calculated scale factor: {final_scale}")
    
    if final_scale < min_scale or final_scale > max_scale:
        print(f"Warning: Scale factor {final_scale} is outside reasonable range [{min_scale}, {max_scale}]")
        # You might want to clamp or exit here. The original code raises ValueError.
    
    # Apply scale
    xyz = np.array([p.xyz for p in points3D.values()]) * final_scale
    rgb = np.array([p.rgb for p in points3D.values()])
    
    new_points3D = {}
    for i, (pid, p) in enumerate(points3D.items()):
        from scene.colmap_loader import Point3D
        new_points3D[pid] = Point3D(id=pid, xyz=xyz[i], rgb=rgb[i],
                                    error=p.error * final_scale, image_ids=p.image_ids,
                                    point2D_idxs=p.point2D_idxs)
                                    
    for img_id in images:
        images[img_id] = images[img_id]._replace(tvec=images[img_id].tvec * final_scale)
        
    # Save
    write_points3D_binary(new_points3D, os.path.join(output_dir, "points3D.bin"))
    write_cameras_binary(cameras, os.path.join(output_dir, "cameras.bin"))
    write_images_binary(images, os.path.join(output_dir, "images.bin"))
    storePly(os.path.join(output_dir, "points3D.ply"), xyz, rgb)
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({"scale": float(final_scale)}, f)
        
    print(f"Rescaled data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--arkit_dir', type=str, required=True)
    parser.add_argument('--min_scale', type=float, default=0.8)
    parser.add_argument('--max_scale', type=float, default=5.0)
    args = parser.parse_args()
    
    calibrate_scale(args.input, args.output, args.arkit_dir, args.min_scale, args.max_scale)
