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


def get_labels(id_list, df_train, df_external):
    train_labels = set()
    external_labels = set()
    for id_str in id_list:
        if id_str in df_train.index:
            labels = df_train.loc[id_str]['Target'].split(' ')
            labels_str = ' '.join(sorted(labels))
            train_labels.add(labels_str)
        elif id_str in df_external.index:
            labels = df_external.loc[id_str]['Target'].split(' ')
            labels_str = ' '.join(sorted(labels))
            external_labels.add(labels_str)
        else:
            print(id_str, 'is not in df_train or df_external')
            return None

    if len(train_labels) == 0:
      return list(external_labels)[0]

    return list(train_labels)[0]


def find_duplicate(hash_func, df_train, df_external,
                   train_filenames, external_filenames):
    images_dict = defaultdict(list)
    for filename in tqdm.tqdm(train_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        images_dict[h].append(filename)
    
    for filename in tqdm.tqdm(external_filenames):
        image = Image.open(filename)
        h = hash_func(image)
        images_dict[h].append(filename)

    records = []
    for key, values in images_dict.items():
        id_list = [os.path.basename(v)[:-len('_green.png')] for v in values]
        values_str = ' '.join(id_list)

        if str(key) == '0000000000000000':
            for id_str in id_list:
                labels = get_labels([id_str], df_train, df_external)
                assert labels is not None
                records.append((key, id_str, labels))
        else:
            labels = get_labels(id_list, df_train, df_external)
            assert labels is not None
            records.append((key, values_str, labels))

    df = pd.DataFrame.from_records(records, columns=['hash', 'Ids', 'Target'])
    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='data', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    raw_images_dir = os.path.join(args.data_dir, 'raw')
    train_dir = os.path.join(raw_images_dir, 'train')
    external_dir = os.path.join(raw_images_dir, 'external')

    train_filenames = list(glob.glob(os.path.join(train_dir, '*_green.png')))
    external_filenames = list(glob.glob(os.path.join(external_dir, '*_green.png')))

    hash_func = {'phash': imagehash.phash,
                 'ahash': imagehash.average_hash}

    df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), index_col='Id')
    df_external = pd.read_csv(os.path.join(args.data_dir, 'external.csv'), index_col='Id')

    for hash_type, hash_func in hash_func.items():
        df_duplicate = find_duplicate(hash_func, df_train, df_external,
                                      train_filenames, external_filenames)
        output_filename = os.path.join(args.data_dir,
                                       'duplicates.{}.csv'.format(hash_type))
        df_duplicate.to_csv(output_filename, index=False)


if __name__ == '__main__':
    main()
