from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import glob
import shutil
from collections import defaultdict

import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import imagehash


def get_labels(filenames, df_train, df_external, h):
    train_id_str_len = len('050a106a-bbc1-11e8-b2bb-ac1f6b6435d0')
    suffix_len = len('_green.png')

    train_labels = set()
    external_labels = set()
    for filename in filenames:
        id_str = os.path.basename(filename)[:-suffix_len]
        if id_str in df_train.index:
            labels = df_train.loc[id_str]['Target'].split(' ')
            labels_str = ' '.join(sorted(labels))
            train_labels.add(labels_str)
        else:
            labels = df_external.loc[id_str]['Target'].split(' ')
            labels_str = ' '.join(sorted(labels))
            external_labels.add(labels_str)

    if len(train_labels) == 0:
      return list(external_labels)[0]

    return list(train_labels)[0]


def find_leak(hash_func, df_train, df_external,
              train_filenames, external_filenames, test_filenames):
    train_dict = defaultdict(list)
    for filename in tqdm.tqdm(train_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        train_dict[h].append(filename)

    for filename in tqdm.tqdm(external_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        train_dict[h].append(filename)

    records = []
    for filename in tqdm.tqdm(test_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        if str(h) == '0000000000000000':
            continue

        if h in train_dict:
            labels = get_labels(train_dict[h], df_train, df_external, h)
            records.append((os.path.basename(filename[:-len('_green.png')]), labels))

    return pd.DataFrame.from_records(records, columns=['Id', 'Target'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='data', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    raw_images_dir = os.path.join(data_dir, 'raw')

    test_dir = os.path.join(raw_images_dir, 'test')
    train_dir = os.path.join(raw_images_dir, 'train')
    external_dir = os.path.join(raw_images_dir, 'external')

    test_filenames = list(glob.glob(os.path.join(test_dir, '*_green.png')))
    train_filenames = list(glob.glob(os.path.join(train_dir, '*_green.png')))
    external_filenames = list(glob.glob(os.path.join(external_dir, '*_green.png')))

    hash_func = {'phash': imagehash.phash,
                 'ahash': imagehash.average_hash}

    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col='Id')
    df_external = pd.read_csv(os.path.join(data_dir, 'external.csv'), index_col='Id')

    for hash_type, hash_func in hash_func.items():
        df_leak = find_leak(hash_func, df_train, df_external,
                            train_filenames, external_filenames, test_filenames)
        output_filename = os.path.join(data_dir, 'data_leak.{}.csv'.format(hash_type))
        df_leak.to_csv(output_filename, index=False)


if __name__ == '__main__':
    main()
