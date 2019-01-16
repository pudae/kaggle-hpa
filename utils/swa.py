# BSD 2-Clause License
#
# Copyright (c) 2018, Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import tqdm

import torch
import torch.nn.functional as F

def adjust_learning_rate(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


def moving_average(net1, net2, alpha=1):
  for param1, param2 in zip(net1.parameters(), net2.parameters()):
    param1.data *= (1.0 - alpha)
    param1.data += param2.data * alpha


def _check_bn(module, flag):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    flag[0] = True


def check_bn(model):
  flag = [False]
  model.apply(lambda module: _check_bn(module, flag))
  return flag[0]


def reset_bn(module):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    module.running_mean = torch.zeros_like(module.running_mean)
    module.running_var = torch.zeros_like(module.running_var)


def _get_momenta(module, momenta):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    momenta[module] = module.momentum


def _set_momenta(module, momenta):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input_dict in tqdm.tqdm(loader):
        input = input_dict['image'].cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def schedule(config, epoch):
  t = epoch / config.swa.start
  lr_ratio = config.swa.lr / config.optimizer.params.lr
  if t <= 0.5:
    factor = 1.0
  elif t <= 0.9:
    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
  else:
    factor = lr_ratio
  return config.optimizer.params.lr * factor


def detach_params(model):
  for param in model.parameters():
    param.detach_()

  return model
