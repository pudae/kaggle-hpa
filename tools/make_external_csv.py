from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--external_images_dir', dest='external_images_dir',
                        help='the directory of the external images',
                        default='data/rgby/external', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    input_filename = 'data/HPAv18RBGY_wodpl.csv'
    output_filename = 'data/external.csv'
    external_images_dir = args.external_images_dir

    ids = [os.path.splitext(fname)[0] for fname in os.listdir(external_images_dir)]
    df = pd.read_csv(input_filename, index_col='Id')

    indices = pd.Index(ids, dtype='object')
    df_new = df.loc[indices]
    df_new.index.name = 'Id'
    df_new.to_csv(output_filename)


if __name__ == '__main__':
    main()
