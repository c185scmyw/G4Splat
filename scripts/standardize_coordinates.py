import os
import sys
import argparse
import numpy as np
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

def standardize_coordinates(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading COLMAP data from {input_dir}")
    cameras = read_cameras_binary(os.path.join(input_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(input_dir, "images.bin"))
    points3D = read_points3D_binary(os.path.join(input_dir, "points3D.bin"))
    
    xyz = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()])
    
    # 1. Center the point cloud
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    
    # 2. PCA for orientation
    pca = PCA(n_components=3)
    pca.fit(xyz_centered)
    
    # Principal components (eigenvectors)
    # components_[0] is the direction of maximum variance (car length)
    # components_[1] is the second maximum (car width)
    # components_[2] is the third (car height)
    rot_matrix = pca.components_ # This is our orientation matrix
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[2] = -rot_matrix[2]
        
    xyz_standard = (rot_matrix @ xyz_centered.T).T
    
    # 3. Align with ego-vehicle conventions
    # Typically: X forward, Y left, Z up
    # After PCA, we might need to flip axes based on camera distribution
    # Calculate camera positions in the new frame
    cam_centers = []
    for img_info in images.values():
        qvec_scipy = img_info.qvec[[1, 2, 3, 0]]
        Rt = np.eye(4)
        Rt[:3, :3] = R.from_quat(qvec_scipy).as_matrix()
        Rt[:3, 3] = img_info.tvec
        C2W = np.linalg.inv(Rt)
        cam_centers.append(C2W[:3, 3])
        
    cam_centers = np.array(cam_centers)
    cam_centers_centered = cam_centers - centroid
    cam_centers_standard = (rot_matrix @ cam_centers_centered.T).T
    
    # Heuristic: Z should be "up". Most cameras are above the ground/car.
    if np.mean(cam_centers_standard[:, 2]) < np.mean(xyz_standard[:, 2]):
        rot_matrix[2] = -rot_matrix[2]
        rot_matrix[1] = -rot_matrix[1] # Keep it right-handed
        xyz_standard = (rot_matrix @ xyz_centered.T).T
        cam_centers_standard = (rot_matrix @ cam_centers_centered.T).T
        
    # Heuristic: X should be "forward". 
    # Usually cameras are more distributed along the length.
    # We can use the first frame or the distribution to decide.
    # For now, let's just use the PCA order and ensure Z is up.
    
    # 4. Final Transform: 
    # Points_new = R @ (Points_old - centroid)
    # Points_new = R @ Points_old - R @ centroid
    
    final_R = rot_matrix
    final_T = -rot_matrix @ centroid
    
    # Apply to points
    new_points3D = {}
    for i, (pid, p) in enumerate(points3D.items()):
        from scene.colmap_loader import Point3D
        new_points3D[pid] = Point3D(id=pid, xyz=xyz_standard[i], rgb=rgb[i],
                                    error=p.error, image_ids=p.image_ids,
                                    point2D_idxs=p.point2D_idxs)
                                    
    # Apply to cameras
    for img_id in images.values():
        # New Pose: P_new = R_new @ X_new + T_new
        # X_new = R @ X_old + T_old_in_new_frame (where T_old_in_new_frame = -R @ centroid)
        # We know: P_old = R_old @ X_old + T_old
        # X_old = R^T @ (X_new - final_T)
        # P_old = R_old @ R^T @ (X_new - final_T) + T_old
        # P_old = (R_old @ R^T) @ X_new + (T_old - R_old @ R^T @ final_T)
        
        R_old = qvec2rotmat(img_info.qvec)
        T_old = img_info.tvec
        
        R_new = R_old @ final_R.T
        T_new = T_old - R_new @ final_T
        
        images[img_id] = img_info._replace(qvec=rotmat2qvec(R_new), tvec=T_new)
        
    # Save
    write_points3D_binary(new_points3D, os.path.join(output_dir, "points3D.bin"))
    write_cameras_binary(cameras, os.path.join(output_dir, "cameras.bin"))
    write_images_binary(images, os.path.join(output_dir, "images.bin"))
    storePly(os.path.join(output_dir, "points3D.ply"), xyz_standard, rgb)
    
    print(f"Standardized data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    standardize_coordinates(args.input, args.output)
