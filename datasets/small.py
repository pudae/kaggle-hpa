from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tqdm
import numpy as np
import pandas as pd
import scipy.misc as misc

import torch
from torch.utils.data.dataset import Dataset


class SmallDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 **_):
        self.split = split
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, 'rgby', 'train')

        self.df_labels = self.load_labels()
        self.examples = self.load_examples()
        self.size = len(self.examples)

    def load_labels(self):
        if self.split == 'train':
            labels_path = os.path.join(self.dataset_dir, 'split.stratified.small.0.csv')
        else:
            labels_path = os.path.join(self.dataset_dir, 'split.stratified.small.1.csv')

        df_labels = pd.read_csv(labels_path)
        df_labels = df_labels[df_labels['Split'] == 'val']

        df_labels['filepath'] = df_labels['Id'].transform(
            lambda v: os.path.join(self.images_dir, v + '.png'))

        return df_labels

    def load_examples(self):
        return [(row['Id'], row['filepath'], [int(l) for l in row['Target'].split(' ')])
                for _, row in self.df_labels.iterrows()]

    def __getitem__(self, index):
        example = self.examples[index]

        filename = example[1]
        image = misc.imread(filename)

        label = [0 for _ in range(28)]
        for l in example[2]:
          label[l] = 1
        label = np.array(label)

        if self.transform is not None:
          image = self.transform(image)

        return {'image': image,
                'label': label,
                'key': example[0]}

    def __len__(self):
        return self.size


def test():
    dataset = SmallDataset('data', 'train', None)
    example = dataset[0]
    print(example['image'].shape, np.where(np.array(example['label']) == 1))
    example = dataset[1]
    print(example['image'].shape, np.where(np.array(example['label']) == 1))

    dataset = SmallDataset('data', 'val', None)
    example = dataset[0]
    print(example['image'].shape, np.where(np.array(example['label']) == 1))
    example = dataset[1]
    print(example['image'].shape, np.where(np.array(example['label']) == 1))

if __name__ == '__main__':
    test()



