from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
from PIL import Image


# from https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984#437386
def download_single_image(i, base_url, save_dir, image_size):
    colors = ['red', 'green', 'blue', 'yellow']
    img_id = i.split('_', 1)
    for color in colors:
        img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
        img_name = i + '_' + color + '.png'

        out_filename = os.path.join(save_dir, img_name)
        if os.path.exists(out_filename):
            continue

        img_url = base_url + img_path

        try:
            # Get the raw response from the url
            r = requests.get(img_url, allow_redirects=True, stream=True)
            r.raw.decode_content = True

            # Use PIL to resize the image and to convert it to L
            # (8-bit pixels, black and white)
            im = Image.open(r.raw)
            im = im.resize(image_size, Image.LANCZOS).convert('L')
            im.save(os.path.join(save_dir, img_name), 'PNG')
        except Exception as e:
            print(e)
            return False

    return True


def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    while len(image_list) > 0:
        print('try to download {} images'.format(len(image_list)))
        try:
            failed_ids = []
            for i in tqdm(image_list, postfix=pid):
                if not download_single_image(i, base_url, save_dir, image_size):
                    failed_ids.append(i)
            image_list = failed_ids
        except Exception as e:
            print(e)


def main():
    # Parameters
    process_num = 24
    image_size = (512, 512)
    url = 'http://v18.proteinatlas.org/images/'
    csv_path =  "data/HPAv18RBGY_wodpl.csv"
    save_dir = "data/raw/external"

    os.makedirs(save_dir, exist_ok=True)

    print('Parent process %s.' % os.getpid())
    img_list = list(pd.read_csv(csv_path)['Id'])
    img_splits = np.array_split(img_list, process_num)
    assert sum([len(v) for v in img_splits]) == len(img_list)
    p = Pool(process_num)
    for i, split in enumerate(img_splits):
        p.apply_async(
            download, args=(str(i), list(split), url, save_dir, image_size)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    main()
