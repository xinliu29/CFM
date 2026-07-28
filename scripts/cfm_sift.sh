#!/bin/bash
NUM_KPTS=(
    512
    1024
    2048
    4096
)


DEVICE_VALUE="0"
DATASET='yfcc'
FEATURE="sift"
USE_NN="True"
USE_PRUNE="False"
FILTER_THR="0.0"
OUTPUT_FILE="cfm_sift.txt"
CHECKPOINT="weights/CFM_sift.pth"

ITER_NUM="1"
VALID_LAYERS="5 8"
USE_FILTER="True"
USE_GLOBAL="True"

# Loop through each checkpoint
for NUM_KPT in "${NUM_KPTS[@]}"; do
    echo "Running evaluation for keypoint number: $NUM_KPT"

    # Execute the Python command with the current checkpoint
    python -m eval.eval_cfm --dataset $DATASET --feature $FEATURE --use_filter $USE_FILTER --use_nn $USE_NN \
    --valid_layers $VALID_LAYERS --iter_num $ITER_NUM \
    --use_prune $USE_PRUNE \
    --weight "$CHECKPOINT" --device $DEVICE_VALUE \
    --filter_thr $FILTER_THR \
    --output_file $OUTPUT_FILE --num_kpt $NUM_KPT --use_global $USE_GLOBAL

    echo "Finished evaluation for keypoint number: $NUM_KPT"
done