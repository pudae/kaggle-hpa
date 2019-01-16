import os
import argparse
import pprint

import numpy as np
import torch
import torch.nn.functional as F

from datasets import get_dataloader
from transforms import get_transform
from models import get_model
import utils.config
import utils.swa as swa
import utils.checkpoint


def get_checkpoints(config, num_checkpoint=10, epoch_end=None):
    checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
    if epoch_end is not None:
        epoch_begin = epoch_end - num_checkpoint + 1
        checkpoints = [os.path.join(checkpoint_dir, 'epoch.{:04d}.pth'.format(e))
                       for e in range(epoch_begin, epoch_end+1)]
        checkpoints = [f for f in checkpoints if os.path.exists(f)]
    else:
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [name for name in checkpoints
                       if name.startswith('epoch') and name.endswith('pth')]
        checkpoints = list(sorted([os.path.join(checkpoint_dir, f) for f in checkpoints]))
        checkpoints = checkpoints[-num_checkpoint:]
    return checkpoints


def run(config, num_checkpoint, epoch_end, output_filename):
    dataloader = get_dataloader(config, 'train', get_transform(config, 'val'))

    model = get_model(config).cuda()
    checkpoints = get_checkpoints(config, num_checkpoint, epoch_end)

    utils.checkpoint.load_checkpoint(model, None, checkpoints[0])
    for i, checkpoint in enumerate(checkpoints[1:]):
        model2 = get_model(config).cuda()
        last_epoch, _ = utils.checkpoint.load_checkpoint(model2, None, checkpoint)
        swa.moving_average(model, model2, 1. / (i + 2))

    with torch.no_grad():
        swa.bn_update(dataloader, model)

    output_name = '{}.{}.{:03d}'.format(output_filename, num_checkpoint, last_epoch)
    print('save {}'.format(output_name))
    utils.checkpoint.save_checkpoint(config, model, None, 0, 0,
                                     name=output_name,
                                     weights_dict={'state_dict': model.state_dict()})


def parse_args():
    parser = argparse.ArgumentParser(description='hpa')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_filename',
                        help='output filename',
                        default='swa', type=str)
    parser.add_argument('--num_checkpoint', dest='num_checkpoint',
                        help='number of checkpoints for averaging',
                        default=10, type=int)
    parser.add_argument('--epoch_end', dest='epoch_end',
                        help='epoch end',
                        default=None, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    
    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    run(config, args.num_checkpoint, args.epoch_end, args.output_filename)
    
    print('success!')


if __name__ == '__main__':
    main()
