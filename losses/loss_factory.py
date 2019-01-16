from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_focal_loss(gamma=2, **_):
    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func


def cross_entropy(**_):
    return torch.nn.BCEWithLogitsLoss()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)
