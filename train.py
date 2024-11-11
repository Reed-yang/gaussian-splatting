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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
transform = transforms.ToTensor()
import cv2
from lpips import LPIPS
from torchvision.utils import save_image

def lpips_loss(loss_fn, pred, gt):
    loss = loss_fn.forward(pred, gt)
    return loss

from torch import nn

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, all_args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, all_args=all_args)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    if all_args.train_ground_extra:
        ground_viewpoint_stack = scene.getGroundCameras().copy()
    # if all_args.use_diffusion_prior:
    #     diffusion_viewpoint_stack = scene.getDiffusionCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    loss_lpips = LPIPS(net="vgg").to("cuda")
    convert_to_buffer(loss_lpips, persistent=False)
    lpips_weight = 1.0
    lpips_start = 5000

    if all_args.conf_path is not None:
        conf_paths = sorted([os.path.join(all_args.conf_path, f) for f in os.listdir(all_args.conf_path) if f.endswith(('.npy'))])
        confs = [torch.tensor(np.load(f), device="cuda") for f in conf_paths]
        # Assuming conf_map is one of the loaded confs with shape (336, 512)
        for i in range(len(confs)):
            conf_map = confs[i]

            # Calculate max and mean values
            max_value = torch.max(conf_map)
            mean_value = torch.mean(conf_map)

            # Calculate padding sizes
            padding_bottom = 512 - conf_map.shape[0]  # Total rows to add
            padding_top = padding_bottom // 2  # Half for the top
            padding_bottom = padding_bottom - padding_top  # Remaining for the bottom

            # Create padding tensors
            top_padding = torch.full((padding_top, conf_map.shape[1]), mean_value, device="cuda")
            bottom_padding = torch.full((padding_bottom, conf_map.shape[1]), max_value, device="cuda")

            # Concatenate to form the final map
            conf_map = torch.cat([top_padding, conf_map, bottom_padding], dim=0)

            # Replace the modified map back in the list
            confs[i] = conf_map

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # use rand to determine ground or not
        ground_prob = all_args.ground_prob
        use_ground = all_args.train_ground_extra and (randint(0, 100) < ground_prob * 100)
        if use_ground:
            if not ground_viewpoint_stack:
                ground_viewpoint_stack = scene.getGroundCameras().copy()
            rand_idx = randint(0, len(ground_viewpoint_stack) - 1)
            viewpoint_cam = ground_viewpoint_stack.pop(rand_idx)
        # ground_extra_start = all_args.ground_extra_start
        # ground_extra_end = all_args.ground_extra_end
        # Pick a random Camera
        else:

        # if all_args.train_ground_extra:
        #     if not ground_viewpoint_stack:
        #         ground_viewpoint_stack = scene.getGroundCameras().copy()
        #     if ground_extra_start is not None:
        #         if iteration < ground_extra_start:
        #             rand_idx = randint(0, len(ground_viewpoint_stack) - 1)
        #             viewpoint_cam = ground_viewpoint_stack.pop(rand_idx)
        #     if ground_extra_end is not None:
        #         if iteration > ground_extra_end:
        #             rand_idx = randint(0, len(ground_viewpoint_stack) - 1)
        #             viewpoint_cam = ground_viewpoint_stack.pop(rand_idx)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            else:
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_idx)
                vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        lo_lp = 0.0
        gt_image = viewpoint_cam.original_image.cuda()
        if 'diffusion' in viewpoint_cam.image_name and all_args.conf_path is not None:
            diff_idx = int(viewpoint_cam.image_name.split('_')[-1].split('.')[0])
            conf_map = confs[diff_idx]
            point_render = f"CityFusionData/ningbo/ningbo_block15/ground/Render/Render_{diff_idx}.png"
            point_render = Image.open(point_render).convert('RGB')
            point_render = transform(point_render).cuda()
            point_render_mask = (point_render.sum(dim=0) > 0).bool()
            # conf_map = F.interpolate(conf_map, size=(image.shape[-2], image.shape[-1]), mode='bilinear')
            conf = (conf_map / torch.max(conf_map)).unsqueeze(0)
            uncertainty = 1 - conf
            # Ll1 = (torch.abs((image - gt_image)) * uncertainty).mean()
            # Ll1 = l1_loss(image * conf, gt_image * conf)
            # rand mask the top area, p = 0.5
            # rand_int = randint(0, 1)
            # top_area_mask = torch.ones_like(gt_image)
            # top_area_mask[:, :gt_image.shape[-1] // 2] = 0.0
            # if rand_int > 0.1:
            top_area_mask = torch.ones((gt_image.shape[-2], gt_image.shape[-1]), device="cuda")
            if viewpoint_cam.use_mask:
                top_area_mask[:gt_image.shape[-1] // 2, :] = 0.0
                gt_image[:,~top_area_mask.bool()] = 0.0
                image[:,~top_area_mask.bool()] = 0.0
            
            Ll1 = l1_loss(image, gt_image)
            # if iteration > lpips_start:
            #     image_for_lp = image
            #     image_for_lp[:,~point_render_mask] = 0.0
            #     # lo_lp = loss_lpips.forward(image_for_lp, point_render).mean()
            #     lo_lp = l1_loss(image_for_lp, point_render) # TODO another l1 loss
                
            

        else:
            Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + lpips_weight * lo_lp

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        if gaussians.sky_distance is not None:
            gaussians.reset_skybox_grad()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and not all_args.disable_adc:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2, 10, 50, 100, 500, 1000, 3000, 4000, 7_000, 10000, 20000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_diffusion_prior", action='store_true', default=False)
    parser.add_argument("--diffusion_path", type=str, default = None)
    parser.add_argument("--use_skybox", action='store_true', default=False)
    parser.add_argument("--skybox_points_num", type=int, default=100_000)
    parser.add_argument("--disable_adc", action='store_true', default=False)
    parser.add_argument("--ground_extra_start", type=int, default=None)
    parser.add_argument("--ground_extra_end", type=int, default=None)
    parser.add_argument("--train_ground_extra", action='store_true', default=False)
    parser.add_argument("--ground_prob", type=float, default=0.5)
    parser.add_argument("--conf_path", type=str, default=None)
    parser.add_argument("--real_data", action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    test_step = args.iterations // 20
    save_step = args.iterations // 10
    args.test_iterations.extend([i for i in range(10000, args.iterations + 1, test_step)])
    args.save_iterations.extend([i for i in range(30000, args.iterations + 1, save_step)])
    args.test_iterations = sorted(set(args.test_iterations))
    args.save_iterations = sorted(set(args.save_iterations))
    # args.iterations = sorted(set(args.iterations))
    print(f"Test iterations: {args.test_iterations}")
    print(f"Save iterations: {args.save_iterations}")
    print(f"Iterations: {args.iterations}")

    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
