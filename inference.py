from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import get_dataloader
from transforms import get_transform
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils.config
import utils.checkpoint


def inference(config, model, split, output_filename=None):
    config.eval.batch_size = 2
    if split == 'test':
        config.data.name = 'TestDataset'

    dataloader = get_dataloader(config, split, get_transform(config, split))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    key_list = []
    label_list = []
    probability_list = []

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)
        for i, data in tqdm.tqdm(enumerate(dataloader), total=total_step):
            images = data['image'].cuda()

            if len(images.size()) == 5:
                B, T, C, H, W = images.size()
                logits = model(images.view(-1, C, H, W))[:,:28]
                logits = logits.view(B, T, -1)
                probabilities = F.sigmoid(logits)
                probabilities = probabilities.mean(dim=1)
            else:
                logits = model(images)[:,:28]
                probabilities = F.sigmoid(logits)

            probability_list.append(probabilities.cpu().numpy())

            if split != 'test':
                label_list.append(data['label'].numpy())

            key_list.extend(data['key'])

        if split != 'test':
            labels = np.concatenate(label_list, axis=0)
            assert labels.ndim == 2
            assert labels.shape[0] == total_size
            assert labels.shape[-1] == 28

        probabilities = np.concatenate(probability_list, axis=0)
        assert probabilities.ndim == 2
        assert probabilities.shape[0] == total_size
        assert probabilities.shape[-1] == 28

        if split != 'test':
            records = []
            for label, probability in zip(labels, probabilities):
              records.append(tuple([str(l) for l in label] + ['{:.04f}'.format(p) for p in probability]))

            columns = ['L{:02d}'.format(l) for l in range(28)] + ['P{:02d}'.format(l) for l in range(28)]
        else:
            records = []
            for key, probability in zip(key_list, probabilities):
                records.append(tuple([key] + ['{:.04f}'.format(p) for p in probability]))

            columns = ['Id'] + ['P{:02d}'.format(l) for l in range(28)]

        df = pd.DataFrame.from_records(records, columns=columns)
        print('save {}'.format(output_filename))
        df.to_csv(output_filename, index=False)


def run(config, split, checkpoint_name, output_filename):
    model = get_model(config).cuda()

    checkpoint = utils.checkpoint.get_checkpoint(config, name=checkpoint_name)
    utils.checkpoint.load_checkpoint(model, None, checkpoint)

    inference(config, model, split, output_filename)


def parse_args():
    parser = argparse.ArgumentParser(description='hpa')
    parser.add_argument('--output', dest='output_filename',
                        help='output filename',
                        default=None, type=str)
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint_filename',
                        help='checkpoint filename',
                        default=None, type=str)
    parser.add_argument('--num_tta', dest='num_tta',
                        help='number of tta images',
                        default=4, type=int)
    parser.add_argument('--split', dest='split',
                        help='split',
                        default='test', type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    torch.multiprocessing.set_sharing_strategy('file_system')

    print('inference HPA')
    args = parse_args()
    config = utils.config.load(args.config_file)
    config.transform.name = 'tta_transform'
    config.transform.params.num_tta = args.num_tta

    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    run(config, args.split, args.checkpoint_filename, args.output_filename)

    print('success!')


if __name__ == '__main__':
  main()
