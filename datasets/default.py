from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.misc as misc

from torch.utils.data.dataset import Dataset


class DefaultDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 split_prefix='split.stratified',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.split_prefix = split_prefix
        self.images_dir = os.path.join(dataset_dir, 'rgby', 'train')
        self.external_images_dir = os.path.join(dataset_dir, 'rgby', 'external')

        self.df_labels = self.load_labels()
        self.examples = self.load_examples()
        self.size = len(self.examples)

    def load_labels(self):
        labels_path = '{}.{}.csv'.format(self.split_prefix, self.idx_fold)
        labels_path = os.path.join(self.dataset_dir, labels_path)
        df_labels = pd.read_csv(labels_path)
        df_labels = df_labels[df_labels['Split'] == self.split]
        df_labels = df_labels.reset_index()

        train_id_len = len('770126a4-bbc6-11e8-b2bc-ac1f6b6435d0')
        def to_filepath(v):
            if len(v) == train_id_len:
                return os.path.join(self.images_dir, v + '.png')
            else:
                return os.path.join(self.external_images_dir, v + '.png')

        df_labels['filepath'] = df_labels['Id'].transform(to_filepath)
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
    dataset = DefaultDataset('data', 'train', None)
    print(len(dataset))
    example = dataset[0]
    example = dataset[1]

    dataset = DefaultDataset('data', 'val', None)
    print(len(dataset))

if __name__ == '__main__':
    test()
