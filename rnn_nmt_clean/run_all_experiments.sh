#!/bin/bash

# ============================================================
# Complete Experiments for RNN-based NMT
# ============================================================
# This script runs all ablation experiments comparing:
# 1. Attention mechanisms: dot, multiplicative, additive
# 2. Training strategies: 100%, 50%, 0% teacher forcing
# 3. Decoding strategies: greedy, beam search
# ============================================================

# Set dataset size (change to '100k' for larger dataset)
DATASET="100k"
GPU_ID=0

# Training parameters
EPOCHS=50
BATCH_SIZE=128
LR=0.0005

echo "================================================================"
echo "RNN-based NMT - Complete Ablation Experiments"
echo "================================================================"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "GPU: $GPU_ID"
echo ""
echo "This will run 5 experiments covering:"
echo "  Part 1: Attention Mechanism Comparison (3 experiments)"
echo "    - dot, multiplicative, additive (with 50% teacher forcing)"
echo ""
echo "  Part 2: Training Strategy Comparison (2 experiments)"
echo "    - 100%, 0% teacher forcing (with additive attention)"
echo ""
echo "  Each experiment will evaluate with multiple beam sizes:"
echo "    - Beam-1 (Greedy), Beam-3, Beam-5, Beam-10"
echo "================================================================"
echo ""

# Save directory
SAVE_DIR="./checkpoints_${DATASET}"
mkdir -p "$SAVE_DIR"

# ============================================================
# Part 1: Attention Mechanism Comparison (50% Teacher Forcing)
# ============================================================

echo "================================================================"
echo "PART 1: Attention Mechanism Comparison"
echo "Teacher Forcing = 50% for all experiments"
echo "================================================================"
echo ""

# ------------------------------------------------------------
# Experiment 1: Dot Product Attention
# ------------------------------------------------------------
echo "====================================="
echo "Exp 1: Dot Product Attention (50% TF)"
echo "====================================="

python train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --attention_type "dot" \
    --teacher_forcing_ratio 0.5 \
    --model_path "${SAVE_DIR}/model_dot_50.pt" \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 1 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 2: Multiplicative Attention
# ------------------------------------------------------------
echo "====================================="
echo "Exp 2: Multiplicative Attention (50% TF)"
echo "====================================="

python train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --attention_type "multiplicative" \
    --teacher_forcing_ratio 0.5 \
    --model_path "${SAVE_DIR}/model_multiplicative_50.pt" \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 2 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 3: Additive Attention
# ------------------------------------------------------------
echo "====================================="
echo "Exp 3: Additive Attention (50% TF)"
echo "====================================="

python train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --attention_type "additive" \
    --teacher_forcing_ratio 0.5 \
    --model_path "${SAVE_DIR}/model_additive_50.pt" \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 3 completed!"
echo ""

# ============================================================
# Part 2: Training Strategy Comparison (Additive Attention)
# ============================================================

echo "================================================================"
echo "PART 2: Training Strategy Comparison"
echo "Using Additive Attention (best from Part 1)"
echo "================================================================"
echo ""

# ------------------------------------------------------------
# Experiment 4: 100% Teacher Forcing
# ------------------------------------------------------------
echo "====================================="
echo "Exp 4: Additive + 100% Teacher Forcing"
echo "====================================="

python train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --attention_type "additive" \
    --teacher_forcing_ratio 1.0 \
    --model_path "${SAVE_DIR}/model_additive_100.pt" \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 4 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 5: 0% Teacher Forcing (Free Running)
# ------------------------------------------------------------
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

# ============================================================
# Summary
# ============================================================

echo "================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "================================================================"
echo ""
echo "Results saved in: $SAVE_DIR/"
echo ""
echo "Trained models:"
echo "  - model_dot_50.pt          (Dot + 50% TF)"
echo "  - model_multiplicative_50.pt (Multiplicative + 50% TF)"
echo "  - model_additive_50.pt      (Additive + 50% TF)"
echo "  - model_additive_100.pt     (Additive + 100% TF)"
echo "  - model_additive_0.pt       (Additive + 0% TF)"
echo ""
echo "Training logs (CSV):"
echo "  - results_dot_50_training_log.csv"
echo "  - results_multiplicative_50_training_log.csv"
echo "  - results_additive_50_training_log.csv"
echo "  - results_additive_100_training_log.csv"
echo "  - results_additive_0_training_log.csv"
echo ""
echo "Next steps:"
echo "  1. Compare BLEU scores from each experiment"
echo "  2. Compare decoding strategies (Greedy vs Beam-3/5/10)"
echo "  3. Plot training curves using CSV logs"
echo "  4. Analyze results in your report"
echo ""
echo "Each results file contains BLEU-4 scores for all beam sizes:"
echo "  - Greedy (Beam-1)"
echo "  - Beam-3"
echo "  - Beam-5"
echo "  - Beam-10"
echo "================================================================"
