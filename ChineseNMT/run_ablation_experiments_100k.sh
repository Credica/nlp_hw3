#!/bin/bash

# ============================================================
# Architecture Ablation Experiments for Transformer NMT - 100K Dataset
# ============================================================
# This script runs architecture ablation experiments on 100k dataset:
# 1. Position Encoding: Absolute vs Relative
# 2. Normalization: LayerNorm vs RMSNorm
#
# Also includes optimizer and label smoothing comparisons (baseline: noamopt=True, smoothing=False)
# ============================================================

# Set common parameters for 100k dataset
DATA_SIZE="100k"
BATCH_SIZE=64
EPOCHS=50
GPU_ID=0

echo "================================================================"
echo "Architecture Ablation Experiments (100K Dataset)"
echo "================================================================"
echo "Using dataset: $DATA_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""
echo "This will run 12 experiments covering:"
echo "  Part 1: Architecture ablation (4 experiments)"
echo "    - Position Encoding: absolute, relative"
echo "    - Normalization: layernorm, rmsnorm"
echo "    (baseline: use_noamopt=True, use_smoothing=False)"
echo ""
echo "  Part 2: Optimizer comparison (4 experiments)"
echo "    - Best architecture with different optimizer/smoothing combinations"
echo ""
echo "  Part 3: Full best config (baseline)"
echo "================================================================"
echo ""

# ============================================================
# Part 1: Architecture Ablation (baseline: noamopt=True, smoothing=False)
# ============================================================

echo "================================================================"
echo "PART 1: Architecture Ablation (NoamOpt=True, Smoothing=False)"
echo "================================================================"
echo ""

# ------------------------------------------------------------
# Experiment 1: Baseline - Absolute Position + LayerNorm
# ------------------------------------------------------------
echo "====================================="
echo "Exp 1: Baseline (Absolute + LayerNorm)"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "layernorm" \
    --exp_name "${DATA_SIZE}_P1_baseline_abs_ln" \
    --use_noamopt \
    --no_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 1 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 2: Relative Position + LayerNorm
# ------------------------------------------------------------
echo "====================================="
echo "Exp 2: Relative Position + LayerNorm"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "layernorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_P1_relative_ln" \
    --use_noamopt \
    --no_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 2 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 3: Absolute Position + RMSNorm
# ------------------------------------------------------------
echo "====================================="
echo "Exp 3: Absolute + RMSNorm"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "rmsnorm" \
    --exp_name "${DATA_SIZE}_P1_abs_rms" \
    --use_noamopt \
    --no_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 3 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 4: Relative Position + RMSNorm (Combined)
# ------------------------------------------------------------
echo "====================================="
echo "Exp 4: Relative + RMSNorm (Combined)"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "rmsnorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_P1_relative_rms" \
    --use_noamopt \
    --no_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 4 completed!"
echo ""

# ============================================================
# Part 2: Optimizer & Smoothing Comparison (using best architecture)
# ============================================================

echo "================================================================"
echo "PART 2: Optimizer & Smoothing Comparison"
echo "Using best architecture from Part 1 (assuming abs+ln as baseline)"
echo "================================================================"
echo ""

# ------------------------------------------------------------
# Experiment 5: NoamOpt + Smoothing=False (baseline, already done in Part1)
# ------------------------------------------------------------
# Skip - already have ${DATA_SIZE}_P1_baseline_abs_ln

# ------------------------------------------------------------
# Experiment 6: NoamOpt + Smoothing=True
# ------------------------------------------------------------
echo "====================================="
echo "Exp 5: NoamOpt + Smoothing=True"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "layernorm" \
    --exp_name "${DATA_SIZE}_P2_noamopt_smooth" \
    --use_noamopt \
    --use_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 5 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 7: NoamOpt=False + Smoothing=False
# ------------------------------------------------------------
echo "====================================="
echo "Exp 6: AdamW + Smoothing=False"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "layernorm" \
    --exp_name "${DATA_SIZE}_P2_adam_nosmooth" \
    --no_noamopt \
    --no_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 6 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 8: NoamOpt=False + Smoothing=True
# ------------------------------------------------------------
echo "====================================="
echo "Exp 7: AdamW + Smoothing=True"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "layernorm" \
    --exp_name "${DATA_SIZE}_P2_adam_smooth" \
    --no_noamopt \
    --use_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 7 completed!"
echo ""

# ============================================================
# Part 3: Best Configuration Combinations
# ============================================================

echo "================================================================"
echo "PART 3: Best Config Combinations"
echo "Testing promising combinations based on research"
echo "================================================================"
echo ""

# ------------------------------------------------------------
# Experiment 9: Relative + LayerNorm + NoamOpt + Smoothing
# ------------------------------------------------------------
echo "====================================="
echo "Exp 8: Rel+LN+Noam+Smooth (Recommended)"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "layernorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_P3_rel_ln_noam_smooth" \
    --use_noamopt \
    --use_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 8 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 10: Relative + RMSNorm + NoamOpt + Smoothing
# ------------------------------------------------------------
echo "====================================="
echo "Exp 9: Rel+RMS+Noam+Smooth"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "rmsnorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_P3_rel_rms_noam_smooth" \
    --use_noamopt \
    --use_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 9 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 11: Relative + LayerNorm + AdamW + No Smoothing
# ------------------------------------------------------------
echo "====================================="
echo "Exp 10: Rel+LN+Adam+NoSmooth"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "relative" \
    --norm_type "layernorm" \
    --max_relative_position 32 \
    --exp_name "${DATA_SIZE}_P3_rel_ln_adam_nosmooth" \
    --no_noamopt \
    --no_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 10 completed!"
echo ""

# ------------------------------------------------------------
# Experiment 12: Absolute + LayerNorm + AdamW + Smoothing
# ------------------------------------------------------------
echo "====================================="
echo "Exp 11: Abs+LN+Adam+Smooth"
echo "====================================="

python main.py \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epoch_num "$EPOCHS" \
    --position_encoding "absolute" \
    --norm_type "layernorm" \
    --exp_name "${DATA_SIZE}_P3_abs_ln_adam_smooth" \
    --no_noamopt \
    --use_smoothing \
    --gpu_id "$GPU_ID"

echo ""
echo "✓ Experiment 11 completed!"
echo ""

# ============================================================
# Summary
# ============================================================

echo "================================================================"
echo "ALL EXPERIMENTS COMPLETED! (100K Dataset)"
echo "================================================================"
echo ""
echo "Results saved in ./experiment/ directory with subfolders:"
echo ""
echo "PART 1 - Architecture Ablation (NoamOpt=True, Smoothing=False):"
echo "  - ./experiment/${DATA_SIZE}_P1_baseline_abs_ln/"
echo "  - ./experiment/${DATA_SIZE}_P1_relative_ln/"
echo "  - ./experiment/${DATA_SIZE}_P1_abs_rms/"
echo "  - ./experiment/${DATA_SIZE}_P1_relative_rms/"
echo ""
echo "PART 2 - Optimizer & Smoothing Comparison:"
echo "  - ./experiment/${DATA_SIZE}_P2_noamopt_smooth/"
echo "  - ./experiment/${DATA_SIZE}_P2_adam_nosmooth/"
echo "  - ./experiment/${DATA_SIZE}_P2_adam_smooth/"
echo ""
echo "PART 3 - Best Config Combinations:"
echo "  - ./experiment/${DATA_SIZE}_P3_rel_ln_noam_smooth/"
echo "  - ./experiment/${DATA_SIZE}_P3_rel_rms_noam_smooth/"
echo "  - ./experiment/${DATA_SIZE}_P3_rel_ln_adam_nosmooth/"
echo "  - ./experiment/${DATA_SIZE}_P3_abs_ln_adam_smooth/"
echo ""
echo "Each folder contains:"
echo "  - model_*.pth       (best trained model)"
echo "  - train_*.log       (training log with BLEU scores)"
echo "  - output_*.txt      (translation outputs)"
echo ""
echo "Next step: Run analyze_results.py to compare performance"
echo "================================================================"
