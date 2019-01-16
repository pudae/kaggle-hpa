#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/resnet34.0.policy.yml
CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/resnet34.1.policy.yml
CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/resnet34.2.policy.yml
CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/resnet34.3.policy.yml
CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/resnet34.4.policy.yml
CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/inceptionv3.attention.policy.per_image_norm.1024.yml
CUDA_VISIBLE_DEVICES=0 python swa.py --config=configs/se_resnext50.attention.policy.per_image_norm.1024.yml

