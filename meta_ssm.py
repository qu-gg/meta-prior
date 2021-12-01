"""
@file metaNN_dgssm.py
@author Ryan Missel

Handles the global version of the MetaPrior for the Bayesian NODE vector field
Inference of the 
"""
import os
import math
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from torch.distributions import Normal, kl_divergence as kl
from utils import Flatten, UnFlatten, get_loader, plot_metric, plot_recon, get_act, parse_args, count_parameters, plot_metaspace_mean


class Network(nn.Module):
    def __init__(self, q, param_amort, z_amort,
                 n_filt, n_hidden_units=100, n_layers=2, dropout=0.0,
                 code_dim=10, layer_norm=True):
        """
        Module that represents the DG-SSM. Has a parameter encoder, q(W|X), and a latent encoder, q(z0|X)

        :param q: size of the latent ODE dimension
        :param param_amort: how many samples to use for parameter encoding
        :param z_amort: how many samples to use for latent position encoding
        :param n_filt: number of filters to start with in the convolution networks
        :param n_hidden_units: number of nodes in the hidden layers of the ODE function
        :param n_layers: number of layers in the ODE function
        :param dropout: rate of dropout on ODE function
        """
        super(Network, self).__init__()

        # Parameters
        self.Q = q
        self.PARAM_AMORT = param_amort
        self.Z_AMORT = z_amort

        # Array that holds dimensions over hidden layers
        self.layers_dim = [q] + n_layers * [n_hidden_units] + [q]
        self.code_dim = code_dim
        self.priorlog = math.log(0.1)

        h_dim = n_filt * 4 ** 3  # encoder output is [4*n_filt,4,4]

        self.layer_norms, self.acts = [], []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.layer_norms.append(nn.LayerNorm(n_out) if layer_norm and i < n_layers else nn.Identity())
            self.acts.append(get_act('tanh') if i < n_layers else get_act('linear'))  # no act. in final layer

        """ MetaPrior """
        self.code_mu = nn.ParameterList([])
        self.code_var = nn.ParameterList([])
        for lidx in range(len(self.layers_dim)):
            code = torch.nn.Parameter(
                torch.zeros([self.layers_dim[lidx], self.code_dim]),
                requires_grad=True
            )
            self.code_mu.append(code)

        for lidx in range(len(self.layers_dim)):
            code = torch.nn.Parameter(
                torch.ones([self.layers_dim[lidx], self.code_dim]),
                requires_grad=True
            )
            self.code_var.append(code)

        self.codes = [1, 2, 3, 4]

        self.hyperprior = nn.Sequential(
            nn.Linear(self.code_dim * 2, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
        )

        self.reset_parameters()

        """ Initial state, z0, encoding """
        self.z_encoder = nn.Sequential(
            nn.Conv2d(z_amort, n_filt, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(n_filt * 2),
            nn.ReLU(),
            nn.Conv2d(n_filt * 2, n_filt * 4, kernel_size=5, stride=2, padding=(2, 2)),
            nn.Tanh(),
            Flatten()
        )

        # Translates encoder embedding into mean and logvar, q(z0|X)
        self.mean_z_net = nn.Sequential(
            nn.Linear(h_dim, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, self.Q),
        )

        self.logvar_z_net = nn.Sequential(
            nn.Linear(h_dim, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, self.Q),
        )

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

        """ Decoder """
        # Translate latent space into decoder ready vector
        self.fc3 = nn.Linear(self.Q, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim // 16, n_filt * 8, kernel_size=4, stride=1, padding=(0, 0)),
            nn.BatchNorm2d(n_filt * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt * 8, n_filt * 4, kernel_size=5, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(n_filt * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt * 4, n_filt * 2, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(n_filt * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt * 2, 1, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Sigmoid(),
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for lidx in range(len(self.layers_dim)):
            # Initialization method of Adv-BNN
            stdv = 1. / math.sqrt(self.code_mu[lidx].size(1))
            self.code_mu[lidx].data.uniform_(-stdv, stdv)
            self.code_var[lidx].data.fill_(self.priorlog)

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        batch_size = self.z_means.shape[0]
        mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])  # N, 2

        q = Normal(mus, logvars.exp())
        N = Normal(torch.zeros(len(mus), device=mus.device),
                   torch.ones(len(mus), device=mus.device))

        klz = kl(q, N).view([batch_size, -1]).sum([1]).mean()
        return klz

    def kl_term(self):
        mus = torch.cat([cmu.view([-1]) for cmu in self.code_mu])
        var = torch.cat([cvar.view([-1]) for cvar in self.code_var])

        kl = self.priorlog - var + (torch.exp(var) ** 2 + (mus - 0) ** 2) / (2 * math.exp(self.priorlog) ** 2) - 0.5
        return kl.sum()

    def draw_f(self):
        """
        Generates the ODE function based on the sampled weights/biases of the sample
        :param idx: idx of the sample in the batch
        :return: generated ODE function
        """
        # Sample the codes array
        codes = [
            self.code_mu[i] + torch.randn_like(self.code_mu[i]) * self.code_var[i].exp()
            for i in range(len(self.layers_dim))
        ]

        def func(x):
            for idx in range(len(self.layers_dim) - 1):
                # Generate weight codes
                temp = codes[idx].unsqueeze(1).repeat(1, self.layers_dim[idx + 1], 1).view([-1, self.code_dim])
                temp2 = codes[idx + 1].unsqueeze(0).repeat(self.layers_dim[idx], 1, 1).view([-1, self.code_dim])
                weight_code = torch.cat((temp2, temp), dim=1)

                # Generate bias codes (concatenation is just with a zeros vector)
                bias_code = torch.cat((torch.zeros_like(codes[idx + 1]), codes[idx + 1]), dim=1)

                # Get weights and biases out
                w = self.hyperprior(weight_code).view([self.layers_dim[idx], self.layers_dim[idx + 1]])
                b = self.hyperprior(bias_code).squeeze()

                # Apply layer with derived weights
                x = self.acts[idx](F.linear(x, w.T, b))

            return x

        return func

    def odefunc(self, t, z, f):
        """ Wrapper function for the odeint calculation """
        z = f(z)
        return z

    def forward(self, x, dt=0.1):
        """
        Forward function of the network. Samples z0, infers W from X, and solves the ODE over the timestep
        before decoding the batch
        Note that z0 has to be iterated over as each sample has its own weights that are used in the solution

        :param x: full sequence of GENERATION length
        :param dt: timescale of differentiation in the odeint function, used as min step size for rk4
        :return: reconstructed sequence Xrec
        """
        batch_size, generation_len = x.shape[0], x.shape[1]
        dim = x.shape[-1]

        # Get q(z0 | X) and sample z0
        h = self.z_encoder(x[:, :self.Z_AMORT].squeeze(2))
        qz0_m, qz0_logv = self.mean_z_net(h), self.logvar_z_net(h)  # N,q & N,q

        eps = torch.randn_like(qz0_m)  # N,q
        z0 = qz0_m + eps * torch.exp(qz0_logv)  # N,q
        self.z_means, self.z_logvs = qz0_m, qz0_logv

        # Evaluate model forward over T to get L latent reconstructions
        t = dt * torch.arange(generation_len, dtype=torch.float).to(z0.device)

        # Draw function for index
        f = self.draw_f()

        # Set odefunc to use
        odef = lambda ts, zs: self.odefunc(ts, zs, f)  # make the ODE forward function

        # Evaluate forward over timestep
        zt = odeint(odef, z0, t, method='rk4', options={'step_size': 0.1})  # [T,q]
        zt = zt.permute([1, 0, 2])

        # Decode
        s = self.fc3(zt.contiguous().view([batch_size * generation_len, z0.shape[1]]))  # L*N*T,h_dim
        Xrec = self.decoder(s)  # L*N*T,nc,d,d
        Xrec = Xrec.view([batch_size, generation_len, 1, dim, dim])  # L,N,T,nc,d,d
        return Xrec


def train_epoch(epoch):
    """
    Handles training the network for a single epoch, generating metric averages and plotting out reconstructions
    every 10 epochs
    :param epoch: current epoch number
    :return: average metrics over epoch
    """
    net.train()
    ts = time.time()
    losses, bce_r_avg, bce_g_avg, klz_avg, klc_avg = [], [], [], [], []

    for i, local_batch in tqdm(enumerate(trainset)):
        optim.zero_grad()

        # Split batch
        _, images, controls, _ = local_batch
        images = images[:, :GENERATION_LEN].to(device)

        # Predict
        preds = net(images)

        # Get loss and update weights
        bce_r, bce_g = bceloss(preds[:, :1], images[:, :1]).sum([2, 3]).view([-1]).mean(), \
                       bceloss(preds[:, 1:], images[:, 1:]).sum([2, 3, 4]).view([-1]).mean()

        # KL terms
        klz, klc = net.kl_z_term(), net.kl_term()

        # Build loss and backprop
        loss = 5 * bce_r + bce_g + Z_BETA * klz + C_BETA * klc
        loss.backward()
        optim.step()

        # Add losses to global arrays
        losses.append(loss.detach().item())
        bce_r_avg.append(bce_r.detach().item())
        bce_g_avg.append(bce_g.detach().item())
        klz_avg.append(klz.detach().item())
        klc_avg.append(klc.detach().item())

    te = time.time()

    # Handle image printing
    if epoch % 10 == 0:
        plot_recon(images, preds, TRAIN_LEN, EXP_NAME, EXP_ID, 'recon{}_train'.format(ep))

    return np.mean(losses), np.mean(bce_r_avg), np.mean(bce_g_avg), \
           np.mean(klz_avg), np.mean(klc_avg), (te - ts) / 60


def test_epoch(epoch):
    """
    Handles testing the DG-SSM at the given epoch on a set of test data
    :param epoch: epoch number
    :return: mean metrics
    """
    net.eval()
    losses, tt_bce_r, tt_bce_g = [], [], []

    with torch.no_grad():
        for i, local_batch in tqdm(enumerate(testset)):
            optim.zero_grad()

            # Split batch
            _, images, controls, _ = local_batch
            images = images[:, :GENERATION_LEN].to(device)

            # Predict
            preds = net(images)
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds)

            # BCE terms for reconstruction and generation respectively
            bce_r, bce_g = bceloss(preds[:, :1], images[:, :1]).sum([2, 3]).view([-1]).mean(), \
                           bceloss(preds[:, 1:], images[:, 1:]).sum([2, 3, 4]).view([-1]).mean()

            # KL terms
            klz, klc = net.kl_z_term(), net.kl_term()

            # Build loss and add to arrays
            loss = bce_r + bce_g + Z_BETA * klz + C_BETA * klc
            losses.append(loss.detach().item())
            tt_bce_r.append(bce_r.detach().item())
            tt_bce_g.append(bce_g.detach().item())

        # Handle image printing
        if epoch % 10 == 0:
            plot_recon(images, preds, TRAIN_LEN, EXP_NAME, EXP_ID, 'recon{}_test'.format(ep))

    return np.mean(losses), np.mean(tt_bce_r), np.mean(tt_bce_g)


def plot_epochs(dataset, tag='train'):
    net.eval()
    with torch.no_grad():
        for i, local_batch in tqdm(enumerate(dataset)):
            optim.zero_grad()

            # Split batch
            _, images, controls, _ = local_batch
            images = images[:, :GENERATION_LEN].to(device)

            # Predict
            preds = net(images)
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds)
            if i > 1:
                break

        # Handle image printing
        plot_recon(images, preds, TRAIN_LEN, EXP_NAME, EXP_ID, 'recon_best_{}.png'.format(tag))


if __name__ == '__main__':
    # Parse and save cmd args
    args = parse_args()

    # fix random seeds for reproducibility
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(123)

    # Device to use for Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    if args.checkpt != 'None':
        exp_path = 'experiments/{}/{}/{}.pth'.format(args.exp_name, args.exp_id, args.checkpt)
        if os.path.isfile(exp_path):
            print("=> loading checkpoint '{}'".format(args.checkpt))
            checkpt = torch.load(exp_path, map_location=device)
            print('checkpoint: ', checkpt.keys())
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpt, checkpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpt))
            exit(0)
    else:
        checkpt = None

    # Hyper-parameters
    EXP_NAME = args.exp_name
    EXP_ID = args.exp_id
    EXP_TYPE = args.exp_type

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    TRAIN_LEN = args.train_len
    GENERATION_LEN = args.generation_len
    DIM = args.dim
    CODE_DIM = args.code_dim

    DROPOUT = args.dropout
    NUM_LAYERS = args.num_layers
    NUM_HIDDEN = args.num_hidden

    PARAM_AMORT = args.param_amort
    Z_AMORT = args.z_amort
    NFILT = args.nfilt
    Q = args.q

    C_BETA = args.c_beta
    Z_BETA = args.z_beta
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    AMSGRAD = args.amsgrad
    LAYER_NORM = args.layer_norm

    # Specify experiment folder
    if not os.path.exists('experiments/{}/{}/'.format(EXP_NAME, EXP_ID)):
        os.makedirs('experiments/{}/{}/'.format(EXP_NAME, EXP_ID))

    if not os.path.exists('experiments/{}/{}/reconstructions/'.format(EXP_NAME, EXP_ID)):
        os.makedirs('experiments/{}/{}/reconstructions/'.format(EXP_NAME, EXP_ID))

    with open('experiments/{}/{}/args.txt'.format(EXP_NAME, EXP_ID), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Initialize network and optimizer
    net = Network(Q, PARAM_AMORT, Z_AMORT, NFILT, n_layers=NUM_LAYERS,
                n_hidden_units=NUM_HIDDEN, dropout=DROPOUT, layer_norm=LAYER_NORM,
                  code_dim=CODE_DIM).to(device)

    # Count parameters and save initial meta-space
    count_parameters(net)
    plot_metaspace_mean(net, EXP_NAME, EXP_ID, train=True)

    epoch_start = 1
    if checkpt is not None:
        net.load_state_dict(checkpt['state_dict'])
        LEARNING_RATE = checkpt['cur_learning_rate']
        epoch_start = checkpt['epoch'] + 1

    # Input generation
    Loader = get_loader(EXP_TYPE)
    trainset = Loader(batch_size=BATCH_SIZE, typ='', num_workers=0)
    testset = Loader(batch_size=BATCH_SIZE, split='test', typ='', num_workers=0)
    print("Training set: ", trainset.shape)

    # Define loss and optimizer
    optim = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=250, gamma=0.85)
    bceloss = nn.BCELoss(reduction='none')
    if checkpt is not None:
        optim.load_state_dict(checkpt['optimizer'])

    # Initialize training loop
    best_bce = np.inf
    tr_losses, tr_bce_rs, tr_bce_gs = [], [], []
    tt_losses, tt_bce_rs, tt_bce_gs = [], [], []
    klzs, klcs = [], []
    not_improved_count = 0

    # Load in saved checkpoint alongside metrics
    if checkpt is not None:
        tr_losses, tt_losses = checkpt['tr_losses'], checkpt['tt_losses']
        tr_bce_rs, tr_bce_gs = checkpt['tr_bce_rs'], checkpt['tr_bce_gs']
        tt_bce_rs, tt_bce_gs = checkpt['tt_bce_rs'], checkpt['tt_bce_gs']
        klzs, klcs = checkpt['klzs'], checkpt['klcs']
        not_improved_count = checkpt['not_improved_count']

    # Train model for number of epochs
    for ep in range(epoch_start, NUM_EPOCHS + 1):
        # Run epoch
        tr_loss, tr_bce_r, tr_bce_g, tr_klz, tr_klc, elapsed_time = train_epoch(ep)
        tt_loss, tt_bce_r, tt_bce_g = test_epoch(ep)

        # Step scheduler and record last learning rate
        sched.step()
        last_lr = sched._last_lr

        # Print averages over epoch
        print("Exp. \'{}_{}\' Epoch {} ({:0.3f}min):".format(EXP_NAME, EXP_ID, ep, elapsed_time))
        print("Train: Loss {:4.3f} - BCE R: {:4.3f} BCE G: {:4.3f} KL Z: {:0.3f} KL C: {:0.3f} ".format(
            tr_loss, tr_bce_r, tr_bce_g, tr_klz, tr_klc
        ))
        print("Test:  Loss {:4.3f} - BCE R: {:4.3f} BCE G: {:4.3f}".format(tt_loss, tt_bce_r, tt_bce_g))

        # Update metric arrays
        tr_losses.append(tr_loss)
        tr_bce_rs.append(tr_bce_r)
        tr_bce_gs.append(tr_bce_g)

        tt_losses.append(tt_loss)
        tt_bce_rs.append(tt_bce_r)
        tt_bce_gs.append(tt_bce_g)

        klzs.append(tr_klz)
        klcs.append(tr_klc)

        # Save metric plots
        plot_metric(EXP_NAME, EXP_ID, 'total_loss', (tr_losses, tt_losses), 'Losses', mse=True)
        plot_metric(EXP_NAME, EXP_ID, 'bce_rec', (tr_bce_rs, tt_bce_rs), 'Training BCE Reconstruction', mse=True)
        plot_metric(EXP_NAME, EXP_ID, 'bce_gen', (tr_bce_gs, tt_bce_gs), 'Training BCE Generation', mse=True)
        plot_metric(EXP_NAME, EXP_ID, 'tr_kl_z', klzs, 'KL Z')
        plot_metric(EXP_NAME, EXP_ID, 'tr_kl_w', klcs, 'KL C')

        log = {
            # Base parameters to reload
            'epoch': ep,
            'state_dict': net.state_dict(),
            'optimizer': optim.state_dict(),
            'cur_learning_rate': last_lr[-1],
            'tr_losses': tr_losses,
            'tt_losses': tt_losses,
            'not_improved_count': not_improved_count,

            # Holding the current arrays for training
            'tr_bce_rs': tr_bce_rs,
            'tr_bce_gs': tr_bce_gs,
            'klzs': klzs,
            'klcs': klcs,

            # Holding the current arrays for testing
            'tt_bce_rs': tt_bce_rs,
            'tt_bce_gs': tt_bce_gs
        }

        # Save the latest model
        torch.save(log, 'experiments/{}/{}/net_latest.pth'.format(EXP_NAME, EXP_ID))

        # Save the model every 10 epochs
        if ep % 50 == 0:
            torch.save(log, 'experiments/{}/{}/net_{}.pth'.format(EXP_NAME, EXP_ID, ep))
            plot_metaspace_mean(net, EXP_NAME, EXP_ID, train=False)

        # Build and save checkpoint
        if tt_loss < best_bce:
            torch.save(log, 'experiments/{}/{}/net_best.pth'.format(EXP_NAME, EXP_ID))
            best_bce = tt_loss
            not_improved_count = 0
            plot_epochs(trainset, 'train')
            plot_epochs(testset, 'test')
        else:
            not_improved_count += 1

        # Early stop if val loss has not improved in so many steps
        if not_improved_count > args.early_stop:
            break
