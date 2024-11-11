from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            for split in ['train', 'test']:
                split_dir = Path(scene_dir) / split

                if not split_dir.is_dir():
                    print(f"Warning: {split_dir} is not a valid directory. Skipping.")
                    continue

                for method in os.listdir(split_dir):
                    method_dir = split_dir / method
                    if not method_dir.is_dir():
                        print(f"Warning: {method_dir} is not a valid directory. Skipping.")
                        continue

                    print("Split: {}, Method: {}".format(split, method))

                    full_dict[scene_dir].setdefault(split, {})[method] = {}
                    per_view_dict[scene_dir].setdefault(split, {})[method] = {}

                    gt_dir = method_dir / "gt"
                    renders_dir = method_dir / "renders"

                    if not gt_dir.is_dir() or not renders_dir.is_dir():
                        print(f"Warning: Missing gt or renders directory in {method_dir}. Skipping.")
                        continue

                    renders, gts, image_names = readImages(renders_dir, gt_dir)

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress for {} - {} - {}".format(scene_dir, split, method)):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")

                    full_dict[scene_dir][split][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                               "PSNR": torch.tensor(psnrs).mean().item(),
                                                               "LPIPS": torch.tensor(lpipss).mean().item()})
                    per_view_dict[scene_dir][split][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                   "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                   "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                    # Free CUDA memory
                    del renders
                    del gts
                    torch.cuda.empty_cache()

                # Save results for each split
                with open(str(split_dir / "results.json"), 'w') as fp:
                    json.dump(full_dict[scene_dir][split], fp, indent=True)
                with open(str(split_dir / "per_view.json"), 'w') as fp:
                    json.dump(per_view_dict[scene_dir][split], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, "due to", str(e))

        # Save the overall results
        with open(str(scene_dir / "comprehensive_results.json"), 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(str(scene_dir / "comprehensive_per_view.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
