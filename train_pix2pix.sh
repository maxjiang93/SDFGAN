#!/bin/bash

# import global variables PROJ_DIR and JOB_NAME
. ./define_path.sh

# variables
JOB_DIR=${PROJ_DIR}/jobs/${JOB_NAME}
DATASET_DIR=${PROJ_DIR}/data
LOG_DIR=${JOB_DIR}/logs
SAMPLE_DIR=${JOB_DIR}/samples
CHECKPOINT_DIR=${JOB_DIR}/checkpoint

# execute
python main.py \
--model=pix2pix \
--is_train \
--epoch=500 \
--batch_size=32 \
--dataset_dir=${DATASET_DIR} \
--log_dir=${LOG_DIR} \
--sample_dir=${SAMPLE_DIR} \
--checkpoint_dir=${CHECKPOINT_DIR} \
--num_gpus=4 \
--dataset=${DATASET}_freqsplit \
--g_learning_rate=0.0005 \
--d_learning_rate=0.0002 \
--gan_weight=1 \
--l1_weight=100 \
--image_depth=64 \
--beta1=0.5 \
