from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim.lr_scheduler as lr_scheduler


def step(optimizer, last_epoch, step_size=80, gamma=0.1, **_):
  return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
  if isinstance(milestones, str):
    milestones = eval(milestones)
  return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                  gamma=gamma, last_epoch=last_epoch)


def exponential(optimizer, last_epoch, gamma=0.995, **_):
  return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


def none(optimizer, last_epoch, **_):
  return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, last_epoch, mode='min', factor=0.1, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, **_):
  return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                        threshold=threshold, threshold_mode=threshold_mode,
                                        cooldown=cooldown, min_lr=min_lr)


def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001, **_):
  print('cosine annealing, T_max: {}, eta_min: {}, last_epoch: {}'.format(T_max, eta_min, last_epoch))
  return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                        last_epoch=last_epoch)


def get_scheduler(config, optimizer, last_epoch):
  func = globals().get(config.scheduler.name)
  return func(optimizer, last_epoch, **config.scheduler.params)

