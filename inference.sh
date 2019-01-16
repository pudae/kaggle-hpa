#!/bin/bash

GPU_IDX=0

#####################################################################################
# resnet34
#####################################################################################
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.0.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.0.test_val.csv \
  --checkpoint=swa.10.040.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.0.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.0.test.csv \
  --checkpoint=swa.10.040.pth \
  --split=test
  
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.1.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.1.test_val.csv \
  --checkpoint=swa.10.040.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.1.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.1.test.csv \
  --checkpoint=swa.10.040.pth \
  --split=test
  
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.2.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.2.test_val.csv \
  --checkpoint=swa.10.040.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.2.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.2.test.csv \
  --checkpoint=swa.10.040.pth \
  --split=test
  
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.3.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.3.test_val.csv \
  --checkpoint=swa.10.040.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.3.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.3.test.csv \
  --checkpoint=swa.10.040.pth \
  --split=test
  
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.4.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.4.test_val.csv \
  --checkpoint=swa.10.040.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/resnet34.4.policy.yml \
  --num_tta=8 \
  --output=inferences/resnet34.4.test.csv \
  --checkpoint=swa.10.040.pth \
  --split=test


#####################################################################################
# inceptionv3
#####################################################################################
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/inceptionv3.attention.policy.per_image_norm.1024.yml \
  --num_tta=8 \
  --output=inferences/inceptionv3.0.test_val.csv \
  --checkpoint=swa.10.027.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/inceptionv3.attention.policy.per_image_norm.1024.yml \
  --num_tta=8 \
  --output=inferences/inceptionv3.0.test.csv \
  --checkpoint=swa.10.027.pth \
  --split=test


#####################################################################################
# se-resnext50
#####################################################################################
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/se_resnext50.attention.policy.per_image_norm.1024.yml \
  --num_tta=8 \
  --output=inferences/se_resnext50.0.test_val.csv \
  --checkpoint=swa.10.022.pth \
  --split=test_val

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
  --config=configs/se_resnext50.attention.policy.per_image_norm.1024.yml \
  --num_tta=8 \
  --output=inferences/se_resnext50.0.test.csv \
  --checkpoint=swa.10.022.pth \
  --split=test
