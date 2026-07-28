#!/bin/bash
DEVICE_VALUE="0"
DATASET='yfcc'
FEATURE="sift"
USE_NN="True"

USE_PRUNE="True"
N_MIN_TOKENS="0.9"
THRESHOLD="-0.8"

LAST_SINKHORN="False"
MATCH_THRESHOLD="0.0"
FILTER_THR="0.0"
OUTPUT_FILE="rpe_yfcc_sift_ecfm.txt"

ITER_NUM="1"
VALID_LAYERS="5 8"
LAYER_PRUNE="True"
USE_FILTER="True"

CHECKPOINT="weights/ECFM_sift.pth"
USE_GLOBAL="True"

python -m eval.eval_cfm --dataset $DATASET --feature $FEATURE --use_filter $USE_FILTER --use_nn $USE_NN \
    --valid_layers $VALID_LAYERS --iter_num $ITER_NUM \
    --use_prune $USE_PRUNE --n_min_tokens $N_MIN_TOKENS --threshold $THRESHOLD \
    --layer_prune $LAYER_PRUNE --weight "$CHECKPOINT" --device $DEVICE_VALUE \
    --last_sinkhorn $LAST_SINKHORN --match_threshold $MATCH_THRESHOLD --filter_thr $FILTER_THR \
    --output_file $OUTPUT_FILE --use_global $USE_GLOBAL
