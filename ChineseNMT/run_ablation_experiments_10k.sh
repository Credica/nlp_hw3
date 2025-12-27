#!/bin/bash

# ============================================================
# Architecture Ablation Experiments for Transformer NMT
# ============================================================
# This script runs all architecture ablation experiments:
# 1. Position Encoding: Absolute vs Relative
# 2. Normalization: LayerNorm vs RMSNorm
# ============================================================

# Set common parameters
DATA_SIZE="10k"  # Use 10k for faster experiments, change to 100k for final results
BATCH_SIZE=32
EPOCHS=40
GPU_ID=0

echo "Starting Architecture Ablation Experiments..."
echo "Using dataset: $DATA_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""

# ============================================================
# Experiment 1: Baseline - Absolute Position + LayerNorm
# ============================================================
echo "====================================="
echo "Experiment 1: Baseline"
echo "  Position Encoding: absolute"
echo "  Normalization: layernorm"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "layernorm" \
    --exp_name "${DATA_SIZE}_baseline_abs_ln" \
    --gpu_id "$GPU_ID"

echo ""
echo "Experiment 1 completed!"
echo ""

# ============================================================
# Experiment 2: Relative Position + LayerNorm
# ============================================================
echo "====================================="
echo "Experiment 2: Relative Position Encoding"
echo "  Position Encoding: relative"
echo "  Normalization: layernorm"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "layernorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_relative_ln" \
    --gpu_id "$GPU_ID"

echo ""
echo "Experiment 2 completed!"
echo ""

# ============================================================
# Experiment 3: Absolute Position + RMSNorm
# ============================================================
echo "====================================="
echo "Experiment 3: RMSNorm"
echo "  Position Encoding: absolute"
echo "  Normalization: rmsnorm"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "rmsnorm" \
    --exp_name "${DATA_SIZE}_abs_rms" \
    --gpu_id "$GPU_ID"

echo ""
echo "Experiment 3 completed!"
echo ""

# ============================================================
# Experiment 4: Relative Position + RMSNorm (Combined)
# ============================================================
echo "====================================="
echo "Experiment 4: Combined (Relative + RMSNorm)"
echo "  Position Encoding: relative"
echo "  Normalization: rmsnorm"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "rmsnorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_relative_rms" \
    --gpu_id "$GPU_ID"

echo ""
echo "Experiment 4 completed!"
echo ""

# ============================================================
# All experiments completed!
# ============================================================
echo "=============================================="
echo "All Architecture Ablation Experiments Completed!"
echo "=============================================="
echo ""
echo "Results saved in ./experiment/ directory:"
echo "  - model_${DATA_SIZE}_baseline_abs_ln.pth"
echo "  - model_${DATA_SIZE}_relative_ln.pth"
echo "  - model_${DATA_SIZE}_abs_rms.pth"
echo "  - model_${DATA_SIZE}_relative_rms.pth"
echo ""
echo "Training logs:"
echo "  - train_${DATA_SIZE}_baseline_abs_ln.log"
echo "  - train_${DATA_SIZE}_relative_ln.log"
echo "  - train_${DATA_SIZE}_abs_rms.log"
echo "  - train_${DATA_SIZE}_relative_rms.log"
echo ""
echo "Next step: Run analyze_results.py to compare performance"
