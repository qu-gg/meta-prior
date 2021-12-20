"""
@file error_metrics.py
@author qu-gg

Handles evaluating benchmark and proposed methods on a given test set to build
per-step error metric plots for comparison
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
from meta_ssm import Network
from skimage.metrics import structural_similarity as ssim
from utils import get_loader, count_parameters, plot_metaspace_mean, plot_expr

# fix random seeds for reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(123)

# Parse and save cmd args
EXP_NAME = 'grav_global'
EXP_ID = '02'
args = json.load(open('experiments/{}/{}/args.txt'.format(EXP_NAME, EXP_ID), 'r'))

# Device to use for Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args['device'])

# Load in checkpoint
exp_path = 'experiments/{}/{}/{}.pth'.format(EXP_NAME, EXP_ID, "net_latest")
if os.path.isfile(exp_path):
    print("=> loading checkpoint '{}'".format("net_latest"))
    checkpt = torch.load(exp_path, map_location=device)
    print('checkpoint: ', checkpt.keys())
    print("=> loaded checkpoint '{}' (epoch {})".format("net_latest", checkpt['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(exp_path))
    exit(0)

# Initialize network and optimizer
net = Network(args['q'], args['param_amort'], args['z_amort'], args['nfilt'], n_layers=args['num_layers'],
              n_hidden_units=args['num_hidden'], dropout=args['dropout'], layer_norm=args['layer_norm'],
              code_dim=args['code_dim']).to(device)

# Count parameters and save initial meta-space
count_parameters(net)
plot_metaspace_mean(net, EXP_NAME, EXP_ID, train=True)

epoch_start = 1
if checkpt is not None:
    net.load_state_dict(checkpt['state_dict'])
    LEARNING_RATE = checkpt['cur_learning_rate']
    epoch_start = checkpt['epoch'] + 1

# Input generation
Loader = get_loader(args['exp_type'])
testset = Loader(batch_size=100, split='test', typ='', shuffle=False, num_workers=0)
print("Testing set: ", testset.shape)

# Get reconstructions from model and stack
net.eval()
preds = None
images = None
with torch.no_grad():
    for i, local_batch in tqdm(enumerate(testset)):
        # Split batch
        _, image, controls, _ = local_batch
        image = image[:, :args['generation_len']].to(device)

        # Predict
        pred = net(image)
        if torch.isnan(pred).any():
            print("NAN encountered")
            pred = torch.nan_to_num(pred)
            continue

        # Stack
        if preds is None:
            preds = pred
            images = image
        else:
            preds = torch.vstack((preds, pred))
            images = torch.vstack((images, image))

# Convert tensors to numpy
preds = preds.detach().cpu().numpy()
images = images.detach().cpu().numpy()


"""
Metric calculations
"""


def mse_calc(x1, x2):
    return np.sum((x1 - x2) ** 2, axis=(-1, -2))


def mae_calc(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=(-1, -2))


def psnr_calc(x1, x2):
    pmse = mse_calc(x1, x2)
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / pmse)
    return psnr


def ssim_calc(xs, ys):
    ssim_mean, ssim_std = [], []
    for t in range(xs.shape[1]):
        # Extract all images at this timestep
        ps, iis = xs[:, t].squeeze(1), ys[:, t].squeeze(1)

        # Iterate over images in timestep and build a list of their ssim
        sm = []
        for p, i in zip(ps, iis):
            sm.append(ssim(p, i, data_range=1))

        ssim_mean.append(np.mean(sm))
        ssim_std.append(np.std(sm))

    return ssim_mean, ssim_std


""" 
Error metrics for Meta-SSM
"""
# MSE
meta_mse = mse_calc(preds, images)
meta_mse_mean = np.mean(meta_mse, axis=0)
meta_mse_std = np.std(meta_mse, axis=0)

# MAE
meta_mae = mae_calc(preds, images)
meta_mae_mean = np.mean(meta_mae, axis=0)
meta_mae_std = np.std(meta_mae, axis=0)

# SSIM generation by per-step aggergation and computation
meta_ssim_mean, meta_ssim_std = ssim_calc(preds, images)

# Peak signal to noise ratio
meta_psnr = psnr_calc(preds, images)
meta_psnr_mean = np.mean(meta_psnr, axis=0)
meta_psnr_std = np.std(meta_psnr, axis=0)


"""
Error metrics for DVBF
"""
dvbf_recon = np.load("data/dvbf_grav_reconstructions.npy")

# MSE
dvbf_mse = mse_calc(dvbf_recon, images)
dvbf_mse_mean = np.mean(dvbf_mse, axis=0)
dvbf_mse_std = np.std(dvbf_mse, axis=0)

# MAE
dvbf_mae = mae_calc(dvbf_recon, images)
dvbf_mae_mean = np.mean(dvbf_mae, axis=0)
dvbf_mae_std = np.std(dvbf_mae, axis=0)


# SSIM generation by per-step aggergation and computation
dvbf_ssim_mean, dvbf_ssim_std = ssim_calc(dvbf_recon, images)

# Peak signal to noise ratio
dvbf_psnr = psnr_calc(dvbf_recon, images)
dvbf_psnr_mean = np.mean(dvbf_psnr, axis=0)
dvbf_psnr_std = np.std(dvbf_psnr, axis=0)


""" 
Plotting 
"""
X = (images * 255).astype(np.uint8)
X1 = (preds * 255).astype(np.uint8)
X2 = (dvbf_recon * 255).astype(np.uint8)

timesteps = images.shape[1]
num_sample = 5
blank = 5
genstart = 10

num_paths = 3

panel = np.ones((32 * num_paths * num_sample + blank * (num_sample + num_paths), 32 * timesteps + num_paths * blank)) * 255
panel = np.uint8(panel)
panel = Image.fromarray(panel)
#
# selected_idx = np.random.choice(X.shape[0], num_sample, replace=False)
# selected_idx = sorted(selected_idx)

imgs = []

for num, ps in enumerate(zip(X, X1, X2)):
    if num > num_sample:
        break

    img = np.zeros((32 * num_paths, (timesteps + 1) * 32)).astype(np.uint8)
    for i in range(genstart):
        for pidx, p in enumerate(ps):
            img[32 * pidx:32 * (pidx + 1), i * 32: (i + 1) * 32] = p[i]

    #
    # img = np.zeros((32 * num_paths, genstart * 32)).astype(np.uint8)
    # for i in range(genstart):
    #     img[:32, i * 32: (i + 1) * 32] = selected_inps[i]
    #     img[32:64, i * 32: (i + 1) * 32] = selected_rcns[i]

    # panel.paste(img, (blank, blank * (num + 1) + num * 32 * num_paths))

    img[:, 32 * genstart:32 * (genstart + 1)] = 255

    for i in range(genstart + 1, timesteps):
        for pidx, p in enumerate(ps):
            img[32 * pidx:32 * (pidx + 1), i * 32: (i + 1) * 32] = p[i]

    img = np.vstack((img, np.zeros([10, (timesteps + 1) * 32])))

    imgs.append(img)
    # # img_gen = np.zeros((32 * 2, (time_steps - genstart) * 32)).astype(np.uint8)
    # # for i in range(time_steps - genstart):
    # #     img_gen[:32, i * 32: (i + 1) * 32] = selected_inps[i + genstart]
    # #     img_gen[32:64, i * 32: (i + 1) * 32] = selected_gens[i]
    #
    # img_gen = Image.fromarray(img_gen)
    # panel.paste(img_gen, (blank * 2 + 32 * genstart, blank * (num + 1) + num * 32 * 2))

img = np.vstack((imgs))
img = Image.fromarray(img)

img.show("DVBF Reconstruction")




plt.axvline(x=9, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.axvline(x=24, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.errorbar(x=range(meta_mse_mean.shape[0]), y=meta_mse_mean.ravel(), yerr=meta_mse_std.ravel())
plt.errorbar(x=range(dvbf_mse_mean.shape[0]), y=dvbf_mse_mean.ravel(), yerr=dvbf_mse_std.ravel())
plt.legend(['', '', 'MetaSSM', 'DVBF'])
plt.title("Per-step MSE")
plt.show()

plt.axvline(x=9, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.axvline(x=24, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.errorbar(x=range(meta_mae_mean.shape[0]), y=meta_mae_mean.ravel(), yerr=meta_mae_std.ravel())
plt.errorbar(x=range(dvbf_mae_mean.shape[0]), y=dvbf_mae_mean.ravel(), yerr=dvbf_mae_std.ravel())
plt.legend(['', '', 'MetaSSM', 'DVBF'])
plt.title("Per-step MAE")
plt.show()

plt.axvline(x=9, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.axvline(x=24, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.errorbar(x=range(len(meta_ssim_mean)), y=meta_ssim_mean, yerr=meta_ssim_std)
plt.errorbar(x=range(len(dvbf_ssim_mean)), y=dvbf_ssim_mean, yerr=dvbf_ssim_std)
plt.legend(['', '', 'MetaSSM', 'DVBF'])
plt.title("Per-step SSIM ")
plt.show()

plt.axvline(x=9, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.axvline(x=24, ymin=0, ymax=30, color='black', linestyle='dotted', linewidth=1)
plt.errorbar(x=range(len(meta_psnr_mean)), y=meta_psnr_mean.ravel(), yerr=meta_psnr_std.ravel())
plt.errorbar(x=range(len(dvbf_psnr_mean)), y=dvbf_psnr_mean.ravel(), yerr=dvbf_psnr_std.ravel())
plt.legend(['', '', 'MetaSSM', 'DVBF'])
plt.title("Per-step PSNR ")
plt.show()