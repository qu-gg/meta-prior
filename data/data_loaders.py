import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from data.boxes import PymunkData
from data.rotmnist import RotMNISTDataset


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


def bouncingball_collate(batch):
    """
    Collate function for the bouncing ball experiments
    Args:
        batch: given batch at this generator call

    Returns: indices of batch, images, controls
    """
    indices, images, controls, classes = [], [], [], []

    for b in batch:
        idx, image, control, cls = b
        indices.append(idx)
        images.append(image)
        controls.append(control)
        classes.append(cls)

    indices = torch.stack(indices)
    images = torch.stack(images)
    controls = torch.stack(controls)
    classes = torch.stack(classes)
    return indices, images, controls, classes


class BoxDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', typ='', split='train', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1):
        assert split in ['train', 'valid', 'test']

        # Generate dataset and initialize loader
        config = {'dataset': 'box', 'out_distr': 'bernoulli', 'dim_u': 1}

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.shape = self.dataset.images.shape
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)


class BoxGravityDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', typ='', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1):
        assert split in ['train', 'valid', 'test']

        # Generate dataset and initialize loader
        config = {'dataset': 'box_gravity', 'out_distr': 'bernoulli', 'dim_u': 1}

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.shape = self.dataset.images.shape
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)


class PolygonDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of polygon ball experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1):
        assert split in ['train', 'valid', 'test']

        # Generate dataset and initialize loader
        config = {'dataset': 'polygon', 'out_distr': 'bernoulli', 'dim_u': 1}

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)


class MixedDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of mixed set of ball experiments
    """
    def __init__(self, batch_size, data_dir='data/box_data', split='train', typ='', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1):
        assert split in ['train', 'valid', 'test', 'holdout']

        # Generate dataset and initialize loader
        config = {'dataset': 'mixed_bound', 'out_distr': 'bernoulli', 'dim_u': 1}

        self.dataset = PymunkData("{}/{}_{}{}.npz".format(data_dir, config['dataset'], split, typ), config)
        self.shape = self.dataset.images.shape
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)


class MixGravityDataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', typ='', split='train', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1):
        assert split in ['train', 'valid', 'test']

        # Generate dataset and initialize loader
        config = {'dataset': 'mixed_gravity', 'out_distr': 'bernoulli', 'dim_u': 1}

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.shape = self.dataset.images.shape
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)


class MixGravity8DataLoader(BaseDataLoader):
    """
    Dataloader for the base implementation of bouncing ball w/ gravity experiments, available here:
    https://github.com/simonkamronn/kvae
    """
    def __init__(self, batch_size, data_dir='data/box_data', typ='', split='train', shuffle=True,
                 collate_fn=bouncingball_collate, num_workers=1):
        assert split in ['train', 'valid', 'test']

        # Generate dataset and initialize loader
        config = {'dataset': 'mixed_gravity_8', 'out_distr': 'bernoulli', 'dim_u': 1}

        self.dataset = PymunkData("{}/{}_{}.npz".format(data_dir, config['dataset'], split), config)
        self.shape = self.dataset.images.shape
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)


""" RotMNIST dataset """


def rotmnist_collate(batch):
    """
    Collate function for the mocap data
    Args:
        batch: given batch at this generator call

    Returns: indices of batch, images, controls
    """
    indices, xs, digits = [], [], []

    for b in batch:
        idx, x, digit = b
        indices.append(idx)
        xs.append(x)
        digits.append(digit)

    indices = torch.stack(indices)
    xs = torch.stack(xs)
    digits = torch.stack(digits)
    return indices, xs, digits


class RotMNISTDataLoader(BaseDataLoader):
    """
    DataLoader responsible for many mocap signals on a single subject 35
    Contains 23 training, 8 val, 5 test
    """
    def __init__(self, batch_size, data_dir='data/base_data/', split='train', digits=[i for i in range(10)],
                 shuffle=True, collate_fn=rotmnist_collate, num_workers=1):
        assert split in ['train', 'valid', 'test']

        # Generate dataset and initialize loader
        self.dataset = RotMNISTDataset(data_dir, digits=digits, split=split)
        self.shape = self.dataset.X.shape
        self.data_dir = data_dir
        self.split = split

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers, collate_fn)
