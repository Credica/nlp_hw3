#!/bin/bash

# 运行实验5
DATASET="100k"
GPU_ID=0
EPOCHS=15
BATCH_SIZE=128
LR=0.0005
SAVE_DIR="./checkpoints_${DATASET}"

echo "====================================="
echo "Exp 5: Additive + 0% Teacher Forcing (Free Running)"
echo "====================================="

python train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --attention_type "additive" \
    --teacher_forcing_ratio 0.0 \
    --model_path "${SAVE_DIR}/model_additive_0.pt" \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 5 completed!"
echo ""
