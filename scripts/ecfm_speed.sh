#!/bin/bash
NUM_KPTS=(
    1000
    2000
    3000
    4000
    5000
    6000
)

DEVICE_VALUE="3"
DATASET='yfcc'
FEATURE="spp"
USE_NN="True"
USE_PRUNE="True"
N_MIN_TOKENS="0.8"
THRESHOLD="-0.9"
FILTER_THR="0.0"
OUTPUT_FILE="ecfm_speed.txt"
CHECKPOINT="weights/ECFM_spp.pth"

ITER_NUM="1"
VALID_LAYERS="5 8"
LAYER_PRUNE="True"
USE_FILTER="True"
USE_GLOBAL="False"

# Loop through each checkpoint
for NUM_KPT in "${NUM_KPTS[@]}"; do
    echo "Running evaluation for keypoint number: $NUM_KPT"

    # Execute the Python command with the current checkpoint
    python -m eval.eval_cfm --dataset $DATASET --feature $FEATURE --use_filter $USE_FILTER --use_nn $USE_NN \
    --valid_layers $VALID_LAYERS --iter_num $ITER_NUM \
    --use_prune $USE_PRUNE --n_min_tokens $N_MIN_TOKENS --threshold $THRESHOLD \
    --layer_prune $LAYER_PRUNE --weight "$CHECKPOINT" --device $DEVICE_VALUE \
    --filter_thr $FILTER_THR \
    --output_file $OUTPUT_FILE --num_kpt $NUM_KPT --use_global $USE_GLOBAL

    echo "Running evaluation for keypoint number: $NUM_KPT"
done