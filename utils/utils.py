from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np

def prepare_train_directories(config):
  out_dir = config.train.dir
  os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)
