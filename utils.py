"""
@file utils.py
@author Ryan Missel

Utility functions across files
"""
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from data.data_loaders import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w ** 2)
        return input.view(input.size(0), nc, self.w, self.w)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='test', help='name of exp to save')
    parser.add_argument('--exp_id', type=str, default='08', help='index of exp to save')
    parser.add_argument('--exp_type', type=str, default='grav', help='which dataset to use')

    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to run over')
    parser.add_argument('--batch_size', type=int, default=25, help='size of batch')
    parser.add_argument('--train_len', type=int, default=10, help='how many X samples to use in model initialization')
    parser.add_argument('--generation_len', type=int, default=25, help='total length to generate')
    parser.add_argument('--dim', type=int, default=32, help='dimension of the image data')
    parser.add_argument('--code_dim', type=int, default=10, help='dimension of the image data')

    parser.add_argument('--bnn', type=bool, default=False, help='whether the ode is a BNN')
    parser.add_argument('--digits', type=list, default=[0, 1, 2, 3, 4, 6], help='which digits to consider in rotMNIST')

    parser.add_argument('--dropout', type=float, default=0.0, help='percent of dropout in the ODE func layers')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the ODE func')
    parser.add_argument('--num_hidden', type=int, default=100, help='number of nodes per hidden layer in ODE func')

    parser.add_argument('--param_amort', type=int, default=10, help='how many X samples to use in parameter inference')
    parser.add_argument('--z_amort', type=int, default=10, help='how many X samples to use in z0 inference')
    parser.add_argument('--nfilt', type=int, default=8, help='number of filters in the CNNs')
    parser.add_argument('--q', type=int, default=32, help='latent dimension size, as well as vector field dimension')

    parser.add_argument('--w_beta', type=float, default=0.01, help='multiplier for params term in loss')
    parser.add_argument('--c_beta', type=float, default=0.0001, help='multiplier for code term in loss')
    parser.add_argument('--z_beta', type=float, default=0.1, help='multiplier for z0 term in loss')

    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay of the optimizer')
    parser.add_argument('--amsgrad', type=bool, default=False, help='whether to use the AMSGrad variant')
    parser.add_argument('--early_stop', type=int, default=200, help='number of steps for early stop')
    parser.add_argument('--layer_norm', type=bool, default=True, help='normalize ODEfunc layers')

    parser.add_argument('--model_type', type=str, default='convODECase', help='which model to evaluate (only effective in evaluation)')

    args = parser.parse_args()
    return args


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "elu":
        return nn.ELU()
    elif act == "celu":
        return nn.CELU()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    else:
        return None


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.ones_like(mu1)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def nll_loss(x_hat, x):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)


def plot_expr(X, Xrec, show=False, fname='rot_mnist.png'):
    """
    Plots reconstructed samples, with ground truth for comparison
    :param X: raw images
    :param Xrec: reconstructed trajectories
    :param show: whether to show it in plt command line
    :param fname: filename
    """
    N = min(X.shape[0], 10)
    Xnp = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    T = X.shape[1]
    plt.figure(2, (T, 3 * N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2 * N, T, i * T * 2 + t + 1)
            plt.imshow(np.reshape(Xnp[i, t], [32, 32]), cmap='gray')
            plt.xticks([])
            plt.yticks([])
        for t in range(T):
            plt.subplot(2 * N, T, i * T * 2 + t + T + 1)
            plt.imshow(np.reshape(Xrecnp[i, t], [32, 32]), cmap='gray')
            plt.xticks([])
            plt.yticks([])
    plt.savefig(fname)
    if show is False:
        plt.close()
    else:
        plt.show()


def plot_metric(enum, eid, metric, values, title, mse=False):
    """ Plotting function for metrics """
    plt.figure(2)
    plt.title(title)
    plt.xlabel("Epoch")

    if mse is True:
        plt.plot(values[0])
        plt.plot(values[1])
        plt.legend(('train', 'test'))
    else:
        plt.plot(values)
    plt.savefig('experiments/{}/{}/{}.png'.format(enum, eid, metric))
    plt.close()


def get_loader(name):
    if name == "box":
        return BoxDataLoader
    elif name == "grav":
        return BoxGravityDataLoader
    elif name == "mixed":
        return MixedDataLoader
    elif name == "mixgrav":
        return MixGravityDataLoader
    elif name == "mixgrav8":
        return MixGravity8DataLoader
    elif name == "singlemocap":
        return SingleMocapDataLoader
    elif name == 'manymocap':
        return ManyMocapDataLoader
    elif name == 'multimocap':
        return MultiMocapDataLoader
    elif name == 'rotmnist':
        return RotMNISTDataLoader
    else:
        return BoxDataLoader


def initialize_weights(m):
  if isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 1)


def plot_recon(X, Xrec, genstart, exp_name, exp_id, fname):
    X = X.detach().cpu().numpy()
    Xrec = Xrec.detach().cpu().numpy()

    [num_sample, time_steps, _, _, _] = X.shape
    blank = 5

    panel = np.ones((32 * 2 * num_sample + blank * (num_sample + 2), 32 * time_steps + 2 * blank)) * 255
    panel = np.uint8(panel)
    panel = Image.fromarray(panel)

    selected_idx = np.random.choice(X.shape[0], num_sample, replace=False)
    selected_idx = sorted(selected_idx)

    for num, idx in enumerate(selected_idx):
        selected_inps = X[idx]
        selected_rcns = Xrec[idx, :genstart]
        selected_gens = Xrec[idx, genstart:]

        selected_inps = np.uint8(selected_inps * 255)
        selected_rcns = np.uint8(selected_rcns * 255)
        selected_gens = np.uint8(selected_gens * 255)

        img = np.zeros((32 * 2, genstart * 32)).astype(np.uint8)
        for i in range(genstart):
            img[:32, i * 32: (i + 1) * 32] = selected_inps[i]
            img[32:64, i * 32: (i + 1) * 32] = selected_rcns[i]

        img = Image.fromarray(img)
        panel.paste(img, (blank, blank * (num + 1) + num * 32 * 2))

        img_gen = np.zeros((32 * 2, (time_steps - genstart) * 32)).astype(np.uint8)
        for i in range(time_steps - genstart):
            img_gen[:32, i * 32: (i + 1) * 32] = selected_inps[i + genstart]
            img_gen[32:64, i * 32: (i + 1) * 32] = selected_gens[i]

        img_gen = Image.fromarray(img_gen)
        panel.paste(img_gen, (blank * 2 + 32 * genstart, blank * (num + 1) + num * 32 * 2))

    panel.save('experiments/{}/{}/reconstructions/{}.png'.format(exp_name, exp_id, fname))


def plot_rot_recon(X, Xrec, genstart, exp_name, exp_id, fname):
    X = X.detach().cpu().numpy()
    Xrec = Xrec.detach().cpu().numpy()

    [num_sample, time_steps, _, _, _] = X.shape
    blank = 5

    panel = np.ones((28 * 2 * num_sample + blank * (num_sample + 2), 28 * time_steps + 2 * blank)) * 255
    panel = np.uint8(panel)
    panel = Image.fromarray(panel)

    selected_idx = np.random.choice(X.shape[0], num_sample, replace=False)
    selected_idx = sorted(selected_idx)

    for num, idx in enumerate(selected_idx):
        selected_inps = X[idx]
        selected_rcns = Xrec[idx, :genstart]
        selected_gens = Xrec[idx, genstart:]

        selected_inps = np.uint8(selected_inps * 255)
        selected_rcns = np.uint8(selected_rcns * 255)
        selected_gens = np.uint8(selected_gens * 255)

        img = np.zeros((28 * 2, genstart * 28)).astype(np.uint8)
        for i in range(genstart):
            img[:28, i * 28: (i + 1) * 28] = selected_inps[i]
            img[28:56, i * 28: (i + 1) * 28] = selected_rcns[i]

        img = Image.fromarray(img)
        panel.paste(img, (blank, blank * (num + 1) + num * 28 * 2))

        img_gen = np.zeros((28 * 2, (time_steps - genstart) * 28)).astype(np.uint8)
        for i in range(time_steps - genstart):
            img_gen[:28, i * 28: (i + 1) * 28] = selected_inps[i + genstart]
            img_gen[28:56, i * 28: (i + 1) * 28] = selected_gens[i]

        img_gen = Image.fromarray(img_gen)
        panel.paste(img_gen, (blank * 2 + 28 * genstart, blank * (num + 1) + num * 28 * 2))

    panel.save('experiments/{}/{}/reconstructions/{}.png'.format(exp_name, exp_id, fname))


def plot_epochs(net, optim, dataset, tag='train'):
    net.eval()
    with torch.no_grad():
        for i, local_batch in tqdm(enumerate(dataset)):
            optim.zero_grad()

            # Split batch
            _, images, controls = local_batch
            images = images[:, :GENERATION_LEN].to(device)

            # Predict
            preds = net(images)
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds)
            if i > 1:
                break

        # Handle image printing
        plot_recon(images, preds, TRAIN_LEN, EXP_NAME, EXP_ID, 'recon_best_{}.png'.format(tag))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_metaspace_mean(net, ename, eid, train=True):
    for c in net.code_mu:
        plt.scatter(c[:, 0].detach().cpu().numpy(), c[:, 1].detach().cpu().numpy())

    plt.legend(('Layer{}'.format(i) for i in range(len(net.codes))))
    plt.title("Meta-variables Mean {} training".format("before" if train else "after"))
    plt.savefig('experiments/{}/{}/metaspace_{}.png'.format(ename, eid, train))
    plt.close()
