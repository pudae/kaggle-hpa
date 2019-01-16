from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=W0611

import os

import tqdm
import numpy as np
import pandas as pd
import scipy.misc as misc

import torch
from torch.utils.data.dataset import Dataset


class TestDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 **_):
        self.split = split
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, 'rgby', 'test')

        self.df_sample = self.load_filenames()
        self.size = len(self.df_sample)

    def load_filenames(self):
        return pd.read_csv(os.path.join(self.dataset_dir, 'sample_submission.csv'))

    def __getitem__(self, index):
        id_str = self.df_sample.iloc[index]['Id']

        filename = os.path.join(self.images_dir, id_str + '.png')
        image = misc.imread(filename)

        if self.transform is not None:
          image = self.transform(image)

        return {'image': image,
                'key': id_str}

    def __len__(self):
        return self.size


def test():
    dataset = DefaultDataset('/data/pudae/hpa/', 'train', None)
    print(len(dataset))
    example = dataset[0]
    example = dataset[1]

    dataset = DefaultDataset('/data/pudae/hpa/', 'val', None)
    print(len(dataset))


if __name__ == '__main__':
    test()

