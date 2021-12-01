"""
@file boxes.py
@author Ryan Missel

Handles generating the datasets for the box experiments under a number of situations
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class PymunkData(Dataset):
    """
    Load sequences of images
    """
    def __init__(self, file_path, config):
        # Load npz files
        npzfile = np.load(file_path)
        clsfile = np.load(file_path[:-4] + '_cls.npz')

        # Load data
        self.images = npzfile['images'].astype(np.float32)
        if config['out_distr'] == 'bernoulli':
            self.images = (self.images > 0).astype('float32')

        self.images = torch.from_numpy(self.images).to(device=torch.Tensor().device)[:1000]
        self.images = self.images.unsqueeze(2)

        # Load classes
        self.classes = torch.from_numpy(clsfile['cls'].astype(np.float32)).to(device=torch.Tensor().device)

        # Load ground truth position and velocity (if present). This is not used in the KVAE experiments in the paper.
        if 'state' in npzfile:
            # Only load the position, not velocity
            self.state = npzfile['state'].astype(np.float32)[:, :, :2]
            self.velocity = npzfile['state'].astype(np.float32)[:, :, 2:]

            # Normalize the mean
            self.state = self.state - self.state.mean(axis=(0, 1))

            # Set state dimension
            self.state_dim = self.state.shape[-1]

        # Get data dimensions
        self.sequences, self.timesteps, self.channels, self.d1, self.d2 = self.images.shape

        # We set controls to zero (we don't use them even if dim_u=1). If they are fixed to one instead (and dim_u=1)
        # the B matrix represents a bias.
        self.controls = np.zeros((self.sequences, self.timesteps, config['dim_u']), dtype=np.float32)
        self.controls = torch.from_numpy(self.controls).to(device=torch.Tensor().device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Couple images and controls together for compatibility with other datasets """
        return torch.Tensor([idx]), self.images[idx], self.controls[idx], self.classes[idx]


if __name__ == '__main__':
    config = {'dataset': 'mixed_test', 'out_distr': 'bernoulli', 'dim_u': 1}

    dataset = PymunkData("box_data/{}.npz".format(config['dataset']), config)
    print(dataset.images.shape)
    print(dataset.images[0].shape)

    images = dataset.images[[0, 20, 25, 4, 5], 0, 0]
    images = np.hstack(images)
    plt.imshow(images, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # for image in dataset.images[np.random.randint(0, dataset.images.shape[0], 1)[0]]:
    #     plt.imshow(image.squeeze())
    #     plt.show()
