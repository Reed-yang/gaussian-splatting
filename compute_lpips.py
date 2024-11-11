import argparse
import lpips
import torch
import torch.nn.functional as F
lpips.LPIPS
class LPIPSWithPerPixelScores(lpips.LPIPS):
    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = []
        for kk in range(self.L):
            if self.lpips:
                per_pixel_score = self.lins[kk](diffs[kk])  # Apply the linear layer
            else:
                per_pixel_score = diffs[kk].sum(dim=1, keepdim=True)  # Sum over channels if not using lpips
            
            # If spatial, upsample to the original image size
            if self.spatial:
                per_pixel_score = F.interpolate(per_pixel_score, size=in0.shape[2:], mode='bilinear', align_corners=False)
            
            res.append(per_pixel_score)

        return torch.cat(res, dim=1)  # Return concatenated per-layer per-pixel scores

# Use argparse to parse command-line arguments as in the original code
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

opt.use_gpu = True
opt.path0 = "diffusion_priors/diffusion_0000.png"
opt.path1 = "Render_0.png"

# Initialize the model with the subclass
loss_fn = LPIPSWithPerPixelScores(net='alex', version=opt.version)

if(opt.use_gpu):
    loss_fn.cuda()

# Load images
img0 = lpips.im2tensor(lpips.load_image(opt.path0))
img1 = lpips.im2tensor(lpips.load_image(opt.path1))

if(opt.use_gpu):
    img0 = img0.cuda()
    img1 = img1.cuda()

# Compute per-pixel similarity scores
dist01 = loss_fn.forward(img0, img1)
print('Per-pixel scores shape:', dist01.shape)  # Print shape to verify per-pixel output
