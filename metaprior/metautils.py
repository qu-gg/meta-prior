"""

"""
import torch
import cmap2d
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_metaspace_mean(net, train=True):
    for c in net.code_mu:
        plt.scatter(c[:, 0].detach().numpy(), c[:, 1].detach().numpy())

    plt.legend(('Layer{}'.format(i) for i in range(len(net.codes))))
    plt.title("Meta-variables Mean {} training".format("before" if train else "after"))
    plt.show()


def plot_metaspace_var(net, train=True):
    for c in net.code_var:
        plt.scatter(c[:, 0].detach().numpy(), c[:, 1].detach().numpy())

    plt.legend(('Layer{}'.format(i) for i in range(len(net.codes))))
    plt.title("Meta-variables Var {} training".format("before" if train else "after"))
    plt.show()


def plot_weight_correlations(net, X, idx, layer):
    # Generate a grid of points with distance h between them
    cmap = cmap2d.SimplexColorMap([[-1, -1], [-1, 1], [1, -1], [1, 1]], [(0, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 0)])

    xx, yy = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
    dots = np.c_[xx.ravel(), yy.ravel()]
    c = [cmap(d) for d in dots]

    weights = None
    for shift in dots:
        _, ws = net.weight_correlations(X, idx, layer, shift)
        if weights is None:
            weights = ws.detach().cpu().numpy().T
        else:
            weights = np.vstack((weights, ws.detach().cpu().numpy().T))

    plt.scatter(weights[:, 0], weights[:, 1], c=c)
    plt.title('Weight Correlations on Adjacent layers for Neuron {}'.format(idx))
    plt.show()


def plot_function_draws(net, X, y, idx, layer):
    # Generate a grid of points with distance h between them
    cmap = cmap2d.SimplexColorMap([[-1, -1], [-1, 1], [1, -1], [1, 1]], [(0, 1, 0), (1, 1, 1), (0, 0, 1), (0, 0, 0)])

    xx, yy = np.meshgrid(np.arange(-1, 1, 0.05), np.arange(-1, 1, 0.05))
    dots = np.c_[xx.ravel(), yy.ravel()]
    colors = [cmap(d) for d in dots]

    sampled_x = torch.linspace(-6, 6, 200)
    for shift, c in zip(dots, colors):
        recons, _ = net.weight_correlations(sampled_x.unsqueeze(1).float(), idx, layer, shift)
        plt.plot(sampled_x, recons.detach().numpy().squeeze(), c=c)

    plt.scatter(X, y, s=1, c='black', marker='o')
    plt.title("Function draws of the MetaPrior BNN, Perturbing One Meta-Var")
    plt.show()
    

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
    elif act == 'logsoftmax':
        return nn.LogSoftmax(dim=1)
    else:
        return None
