import open3d as o3d
import numpy as np
pcd_path = "data/ningbo_block0/flight/sparse/0/points3D_ori.ply"
pcd_down_path = "data/ningbo_block0/flight/sparse/0/points3D.ply"
pcd = o3d.io.read_point_cloud(pcd_path)
pcd = pcd.voxel_down_sample(voxel_size=1.1)
o3d.io.write_point_cloud(pcd_down_path, pcd)
print(f"Have downsampled {pcd_path} to {len(pcd.points)} points.")