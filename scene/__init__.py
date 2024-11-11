#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch import Tensor
from torch import nn
import math
import numpy as np
from scene.dataset_readers import storePly

def camera_project(cameras: list, xyz: Tensor) -> Tensor:
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    origins = cameras.poses[..., :3, 3]
    rotation = cameras.poses[..., :3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_models, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx, fy, cx, cy = torch.moveaxis(cameras.intrinsics, -1, 0)
    x = fx * x + cx
    y = fy * y + cy
    return torch.stack((x, y), -1)

def get_uniform_points_on_sphere_fibonacci(num_points, *, dtype=None, xnp=torch):
    # https://arxiv.org/pdf/0912.4540.pdf
    # Golden angle in radians
    if dtype is None:
        dtype = xnp.float32
    phi = math.pi * (3. - math.sqrt(5.))
    N = (num_points - 1) / 2
    i = xnp.linspace(-N, N, num_points, dtype=dtype)
    lat = xnp.arcsin(2.0 * i / (2*N+1))
    lon = phi * i

    # Spherical to cartesian
    x = xnp.cos(lon) * xnp.cos(lat)
    y = xnp.sin(lon) * xnp.cos(lat)
    z = xnp.sin(lat)
    return xnp.stack([x, y, z], -1)

def get_sky_points(num_points, points3D: Tensor, cameras: list):
    # TODO now dont support cull points by cameras
    # TODO 先用tensor，方便实现，后续调整为无需反复在numpy和torch之间转换
    xnp = torch
    points3D = torch.Tensor(points3D)
    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=xnp)
    points = points.to(points3D.device)
    mean = points3D.mean(0)[None]
    sky_distance = xnp.quantile(xnp.linalg.norm(points3D - mean, 2, -1), 0.97) * 10
    points = points * sky_distance
    points = points + mean
    # gmask = torch.zeros((points.shape[0],), dtype=xnp.bool, device=points.device)
    # for cam in tqdm(cameras, desc="Generating skybox"):
    #     uv = camera_project(cam, points[xnp.logical_not(gmask)])
    #     mask = xnp.logical_not(xnp.isnan(uv).any(-1))
    #     # Only top 2/3 of the image
    #     assert cam.image_sizes is not None
    #     mask = xnp.logical_and(mask, uv[..., -1] < 2/3 * cam.image_sizes[..., 1])
    #     gmask[xnp.logical_not(gmask)] = xnp.logical_or(gmask[xnp.logical_not(gmask)], mask)
    # return points[gmask], sky_distance / 2
    return points, sky_distance / 2


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, all_args = None, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.ground_cameras = {}
        self.diffusion_cameras = {}
        self.diffusion_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "aerial.json")):
            print("Found aerial.json file, assuming CityFusion data set!")
            scene_info = sceneLoadTypeCallbacks["CityFusion"](args.source_path, args.images, args.eval, all_args)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print(f"Lenght of train_cameras: {len(self.train_cameras[resolution_scale])}")
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
            print(f"Lenght of test_cameras: {len(self.test_cameras[resolution_scale])}")
            if all_args is not None and all_args.train_ground_extra:
                print("Loading Ground Cameras")
                self.ground_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.ground_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
                print(f"Lenght of ground_cameras: {len(self.ground_cameras[resolution_scale])}")
            if all_args is not None and all_args.use_diffusion_prior:
                print("Loading Diffusion Cameras")
                image_size = (self.train_cameras[1][0].image_width, self.train_cameras[1][0].image_height)
                self.diffusion_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.diffusion_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True, size=image_size)
                print(f"Lenght of diffusion_cameras: {len(self.diffusion_cameras[resolution_scale])}")


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # self.sky_distance = None
            if all_args.use_skybox:
                scene_pc = scene_info.point_cloud.points
                print(f"Num of scene points: {scene_pc.shape[0]}")
                num_sky_pc = 100000
                sky_pc, sky_distance = get_sky_points(num_sky_pc, scene_pc, self.train_cameras[1.0])
                sky_pc, sky_distance = sky_pc.cpu().numpy(), sky_distance.cpu().numpy()
                gaussians.sky_distance = sky_distance
                skycolor = np.array([[237,247,252]], dtype=np.uint8).repeat(sky_pc.shape[0], axis=0)
                skycolor = skycolor.astype(np.float32) / 255.0
                # TODO set sky pc opacity to 1
                # opacities = np.concatenate((opacities, 1.0 * np.ones(skybox.shape[0])))
                scene_info.point_cloud.points = np.concatenate((scene_pc, sky_pc))
                scene_info.point_cloud.colors = np.concatenate((scene_info.point_cloud.colors, skycolor))
                scene_info.point_cloud.normals = np.zeros_like(scene_info.point_cloud.points)
                print(f"Num of sky points: {sky_pc.shape[0]}")
            self.gaussians.create_from_pcd(all_args, scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getGroundCameras(self, scale=1.0):
        if self.ground_cameras is None:
            return None
        return self.ground_cameras[scale]
    
    def getDiffusionCameras(self, scale=1.0):
        if self.diffusion_cameras is None:
            return None
        return self.diffusion_cameras[scale]
