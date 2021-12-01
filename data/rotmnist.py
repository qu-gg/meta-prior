"""
@file rotmnist.py
@author Ryan Missel

Dataloader for the rotating MNIST dataset
"""
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class RotMNISTDataset(Dataset):
    def __init__(self, data_dir, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], split='train', plot=False):
        # Load in the mat files and extract X data
        fullname = os.path.join(data_dir, "rot-mnist.mat")
        data = sio.loadmat(fullname)
        Xread = np.squeeze(data['X'])
        Digits = np.squeeze(data['Y'])

        indices = np.where(np.isin(Digits, digits))[0]
        Xread, Digits = Xread[indices], Digits[indices]

        N = np.shape(Xread)[0]
        M = N // 10

        # Split into relevant sets
        tr_idx = np.arange(0, N - 2 * M)
        Xtr = Xread[tr_idx, :, :]
        print(Xtr.shape)

        val_idx = np.arange(N - 2 * M, N - M)
        Xval = Xread[val_idx, :, :]

        test_idx = np.arange(N - M, N)
        Xtest = Xread[test_idx, :, :]
        print(Xtest.shape)

        # Choose split
        if split == "train":
            self.X = torch.from_numpy(Xtr).float()
        elif split == "valid":
            self.X = torch.from_numpy(Xval).float()
        elif split == "test":
            self.X = torch.from_numpy(Xtest).float()

        self.X = self.X.to(device=torch.Tensor().device).reshape(self.X.shape[0], self.X.shape[1], 1, 28, 28)
        self.classes = torch.full([self.X.shape[0]], 0)

        # Plot some examples if needed
        if plot:
            x = self.X[np.random.randint(0, 700, 7)]
            plt.figure(1, (20, 8))
            for j in range(6):
                for i in range(16):
                    plt.subplot(7, 20, j * 20 + i + 1)
                    plt.imshow(np.reshape(x[j, i, :], [28, 28]), cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
            plt.show()
            plt.close()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """ Couple images and controls together for compatibility with other datasets """
        return torch.Tensor([idx]), self.X[idx], self.classes[idx]


if __name__ == '__main__':
    d = RotMNISTDataset('base_data/', digits=[0,1,2,3,4,6])
