import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import re

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    aerial_cameras: list = None
    ground_cameras: list = None
    diffusion_cameras: list = None

class CameraView:
    def __init__(self, view_id, image_path, depth_path, position, orientation, width, height, fov):
        self.view_id = view_id
        self.image_path = image_path
        self.depth_path = depth_path
        self.position = position
        self.orientation = orientation
        self.width = width
        self.height = height
        self.fov = fov

class CameraShot:
    def __init__(self, shot_id):
        self.shot_id = shot_id
        self.views = []

    def add_view(self, view):
        self.views.append(view)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def airsim2opencv(qevc, tevc):
    qw, qx, qy, qz = qevc
    x, y, z = tevc
    T = np.eye(4)
    T[:3,3] = [-y, -z, -x]
    
    R = np.eye(4)
    R[:3,:3] = quaternion_to_rotation_matrix([qw, qy, qz, qx]) # TODO check order here
    
    C = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

    F = R.T @ T @ C
    c2w = np.linalg.inv(F)
    return c2w

def readCityFusionCameras(aerial_views, ground_views, eval=False):
    aerial_cam_infos = []
    print(f"Reading {len(aerial_views)} aerial views")
    for i, view in enumerate(aerial_views):
        image = Image.open(view.image_path)
        width, height, fov = view.width, view.height, view.fov
        # FovY = FovX = fov
        fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
        # cx = width / 2
        # cy = height / 2
        # K = np.array([
        #     [fx, 0, cx],
        #     [0, fy, cy],
        #     [0, 0, 1]
        # ])
        FovX = FovY = focal2fov(fx, width)

        # extract pose in airsim coor.
        c2w = airsim2opencv(view.orientation, view.position)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # uid set to be loop index i
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=view.image_path, image_name=view.image_path, width=width, height=height,
                                depth_params=None, depth_path="", is_test=False)
        aerial_cam_infos.append(cam_info)
    
    ground_cam_infos = []
    print(f"Reading {len(ground_views)} ground views")
    for view in ground_views:
        image = Image.open(view.image_path)
        width, height, fov = view.width, view.height, view.fov
        # FovY = FovX = fov
        # extract pose in airsim coor.
        c2w = airsim2opencv(view.orientation, view.position)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
        FovX = FovY = focal2fov(fx, width)

        # ground view_id is unique
        cam_info = CameraInfo(uid=view.view_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=view.image_path, image_name=view.image_path, width=width, height=height,
                                depth_params=None, depth_path="", is_test=eval)
        ground_cam_infos.append(cam_info)

    return aerial_cam_infos, ground_cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    # TODO 调整为在fetchPly中添加sky box
    
    # TODO support downsampling
    if positions.shape[0] > 1e7:
        print("Downsampling point cloud")
        idx = np.random.choice(positions.shape[0], int(8e6), replace=False)
        positions = positions[idx]
        colors = colors[idx]
        normals = normals[idx]
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def path_replace(camera_views):
    # 定义用于替换路径的通用函数
    def re_substitute(pattern, repl_func, path):
        return re.sub(pattern, repl_func, path)

    def replace_depth(path, depth_start=None):
        # DepthVis 替换逻辑
        def replace_depthvis(match):
            prefix, part1, part2 = match.groups()
            return f"{prefix}/flight/DepthPerspective/image_{int(part1)+depth_start}_{part2}_2.pfm"
        
        pattern = r'(.*)/DepthVis/DepthVis_(\d+)_(\d+).png'
        return re_substitute(pattern, replace_depthvis, path)

    def replace_image(path):
        # Scene 替换逻辑
        def replace_image_re(match):
            prefix, part1 = match.groups()
            return f"{prefix}/flight/Scene/{part1}"
    
        pattern = r'(.*)/Scene/(.*)'
        return re_substitute(pattern, replace_image_re, path)

    recomputed_depth = False
    depth_start = 0
    if recomputed_depth:
        # 重新计算深度文件名
        view_start = camera_views[0]
        depth_path_p = os.path.dirname(os.path.dirname(view_start.depth_path))
        depth_path = os.path.join(depth_path_p, "flight", "DepthPerspective")
        depth_paths = sorted(os.listdir(depth_path))
        depth_start = int(depth_paths[0].split('_')[1])

    # 遍历并更新视图路径
    for view in camera_views:
        view.image_path = replace_image(view.image_path)
        view.depth_path = replace_depth(view.depth_path, depth_start=depth_start)

def load_aerial_views(path, json_data):
    # TODO re-implement this to support ground-new json
    # Construct camera shots, views, images, and related intrinsics data
    camera_shots = []
    shot_id = 0
    data_root = os.path.dirname(os.path.dirname(path))
    for shot_data in json_data['data']:
        camera_shot = CameraShot(shot_id)
        view_id = 0
        for view in shot_data['exposure']:
            image_path = os.path.join(data_root, view['image'])
            depth_path = os.path.join(data_root, view['depth'])
            if not (os.path.exists(depth_path) and os.path.exists(image_path)):
                assert False, f"Image or depth file not found: {image_path} or {depth_path}"
            position = view['position']
            orientation = view['orientation']
            hwf = view['hwf']
            width, height, fov = hwf[0], hwf[1], hwf[2]
            camera_view = CameraView(view_id, image_path, depth_path, position, orientation, width, height, fov)
            camera_shot.add_view(camera_view)
            view_id = view_id + 1 # for next view

        camera_shots.append(camera_shot)
        shot_id = shot_id + 1 # for next shot

    # Flatten camera views for saving, all views
    camera_views = [view for shot in camera_shots for view in shot.views]
    return camera_views

def load_ground_views(path, json_data): 
    # ground has no shots, all views
    camera_views = []
    view_id = 0
    data_root = os.path.dirname(os.path.dirname(path))
    data_split_name = path.split('/')[-2] # like 'ningbo'
    for view in json_data['data']:
        image_path = os.path.join(data_root, view['image'])
        depth_path = os.path.join(data_root, data_split_name, view['image'])
        if not os.path.exists(image_path): # TODO now, dont check depth file, depth path can be None also
            assert False, f"Image file not found: {image_path}"
        if not os.path.exists(depth_path):
            depth_path = None
        position = view['position']
        orientation = view['orientation']
        hwf = view['hwf']
        width, height, fov = hwf[0], hwf[1], hwf[2]
        aerial_index = view['aerial_index']
        view_index = view['view_index']
        pointrender_path = os.path.join(data_root, view['render'])
        camera_view = CameraView(view_id, image_path, depth_path, position, orientation, width, height, fov)
        camera_views.append(camera_view)
        view_id = view_id + 1
    return camera_views

def readCityFusionSceneInfo(path, images, eval, all_args, llffhold=8):
    # read and compute cameras from json
    aerial_json = read_json_file(os.path.join(path, "aerial.json"))
    if os.path.exists(os.path.join(path, "ground_new.json")):
        ground_json = read_json_file(os.path.join(path, "ground_new.json"))
    else:
        ground_json = None

    aerial_views = load_aerial_views(path, aerial_json)
    if ground_json:
        ground_views = load_ground_views(path, ground_json)
    else:
        ground_views = []

    aerial_cam_infos, ground_cam_infos = readCityFusionCameras(aerial_views, ground_views, eval=eval)
    all_cam_infos = aerial_cam_infos + ground_cam_infos
    # TODO new style to determine train/test
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # if eval:
    #     train_cam_infos = aerial_cam_infos
    #     test_cam_infos = ground_cam_infos
    # else:
    #     train_cam_infos = aerial_cam_infos + ground_cam_infos 
    #     test_cam_infos = []
    train_cam_infos = [c for c in all_cam_infos if not c.is_test]
    test_cam_infos = [c for c in all_cam_infos if c.is_test]

    # load diffusion prior
    diffusion_cam_infos = []
    if all_args and all_args.use_diffusion_prior:
        diffusion_cam_infos = []
        diffusion_img_path = all_args.diffusion_path
        diffusion_paths = sorted(os.listdir(diffusion_img_path))
        # in cam info, 5 for image, 7 for image path, 8 for image name
        for cam in ground_cam_infos:
            image = Image.open(os.path.join(diffusion_img_path, diffusion_paths[cam.uid]))
            image_path = os.path.join(diffusion_img_path, diffusion_paths[cam.uid])
            image_name = diffusion_paths[cam.uid]
            cam_info = CameraInfo(uid=cam.uid, R=cam.R, T=cam.T, FovY=cam.FovY, FovX=cam.FovX, image=image,
                                image_path=image_path, image_name=image_name, width=cam.width, height=cam.height,
                                depth_params=None, depth_path="", is_test=False)
            diffusion_cam_infos.append(cam_info)
        train_cam_infos += diffusion_cam_infos
        ground_cam_infos = diffusion_cam_infos
        
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # TODO check pcd name here, fuqiang name it `point3D`
    # instead of normally `points3D`
    ply_path = os.path.join(path, "pointclouds/point3D.ply") 

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           aerial_cameras=aerial_cam_infos,
                           ground_cameras=ground_cam_infos,
                           diffusion_cameras=diffusion_cam_infos)
    return scene_info