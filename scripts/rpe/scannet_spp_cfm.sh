#!/bin/bash
DEVICE_VALUE="0"
DATASET='scannet'
FEATURE="spp"
USE_NN="False"

USE_PRUNE="False"

LAST_SINKHORN="True"
MATCH_THRESHOLD="0.0"
FILTER_THR="0.0"
OUTPUT_FILE="rpe_scannet_spp_cfm.txt"

ITER_NUM="1"
VALID_LAYERS="5 8"
USE_FILTER="True"
USE_POSELIB="True"

CHECKPOINT="weights/CFM_spp.pth"

python -m eval.eval_cfm --dataset $DATASET --feature $FEATURE --use_filter $USE_FILTER --use_nn $USE_NN \
    --filter_thr $FILTER_THR --valid_layers $VALID_LAYERS --iter_num $ITER_NUM \
    --use_prune $USE_PRUNE --use_poselib $USE_POSELIB \
    --weight "$CHECKPOINT" --device $DEVICE_VALUE --output_file $OUTPUT_FILE \
    --last_sinkhorn $LAST_SINKHORN --match_threshold $MATCH_THRESHOLD
    
