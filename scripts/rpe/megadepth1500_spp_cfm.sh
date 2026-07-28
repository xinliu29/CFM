#!/bin/bash
# Set variables for command-line argument values
DEVICE_VALUE="0"
DATASET='megadepth1500'
FEATURE="spp"
USE_NN="True"

USE_PRUNE="False"

FILTER_THR="0.0"
OUTPUT_FILE="rpe_megadepth1500_spp_cfm.txt"

ITER_NUM="1"
VALID_LAYERS="5 8"
LAYER_PRUNE="True"
USE_FILTER="True"

CHECKPOINT="weights/CFM_spp.pth"

python -m eval.eval_cfm --dataset $DATASET --feature $FEATURE --use_filter $USE_FILTER --use_nn $USE_NN \
    --valid_layers $VALID_LAYERS --iter_num $ITER_NUM \
    --use_prune $USE_PRUNE \
    --layer_prune $LAYER_PRUNE --weight "$CHECKPOINT" --device $DEVICE_VALUE \
    --filter_thr $FILTER_THR \
    --output_file $OUTPUT_FILE
