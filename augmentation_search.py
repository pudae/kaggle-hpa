from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pprint
import random

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from datasets import get_dataloader
from transforms import get_transform, POLICIES
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils
import utils.config
import utils.checkpoint

from train import train


def sample_policy():
    def sample_params(r):
        if isinstance(r, list):
            return random.choice(r)

        if not isinstance(r, tuple):
            return r

        r1, r2 = r
        if not isinstance(r1, tuple):
            assert not isinstance(r2, tuple)
            if isinstance(r1, float):
                return random.uniform(r1, r2)
            else:
                return random.randint(r1, r2)

        assert isinstance(r1, tuple)
        assert isinstance(r2, tuple)
        return (sample_params(r1), sample_params(r2))

    policies = []

    for _ in range(5):
        policy_0, policy_1 = random.sample(POLICIES, 2)
        params_0 = {key:sample_params(value) for key, value in policy_0[1].items()}
        params_1 = {key:sample_params(value) for key, value in policy_1[1].items()}
        policies.append(((policy_0[0], params_0), (policy_1[0], params_1)))
    return policies


def search_once(config, policy):
    model = get_model(config).cuda()
    criterion = get_loss(config)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer, -1)

    transforms = {'train': get_transform(config, 'train', params={'policies': policy}),
                  'val': get_transform(config, 'val')}
    dataloaders = {split:get_dataloader(config, split, transforms[split])
                   for split in ['train', 'val']}

    score_dict = train(config, model, dataloaders, criterion, optimizer, scheduler, None, 0)
    return score_dict['f1_mavg']


def run(config):
    train_dir = config.train.dir
    writer = SummaryWriter(config.train.dir)
    utils.prepare_train_directories(config)

    # base_policy
    policy = []
    score = search_once(config, policy)
    print('===============================')
    print('base score:', score)
    writer.add_scalar('val/f1', score, 0)

    policies = []
    for i in range(50):
        policy = sample_policy()
        score = search_once(config, policy)
        writer.add_scalar('val/f1', score, i+1)
        policies.append((score, policy))
        policies = list(sorted(policies, key=lambda v: v[0]))[-5:]

        with open(os.path.join(config.train.dir, 'best_policy.data'), 'w') as fid:
            fid.write(str([v[1] for v in policies]))

        for score, policy in policies:
            print('score:', score)
            print('policy:', policy)


def parse_args():
    parser = argparse.ArgumentParser(description='HPA')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('Search Augmentation!!')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
