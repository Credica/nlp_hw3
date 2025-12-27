#!/bin/bash

# ============================================================
# Complete Ablation Experiments for Transformer NMT - 100K Dataset
# ============================================================
# This script runs ALL ablation experiments on 100k dataset:
# 1. Position Encoding: Absolute vs Relative
# 2. Normalization: LayerNorm vs RMSNorm
# 3. Optimizer: NoamOpt vs AdamW (use_noamopt)
# 4. Label Smoothing: True vs False (use_smoothing)
#
# Total experiments: 2 x 2 x 2 x 2 = 16 combinations
# ============================================================

# Set common parameters for 100k dataset
DATA_SIZE="100k"
BATCH_SIZE=64
EPOCHS=50
GPU_ID=0

echo "================================================================"
echo "Complete Ablation Experiments (100K Dataset)"
echo "================================================================"
echo "Using dataset: $DATA_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""
echo "This will run 16 experiments covering all combinations of:"
echo "  - Position Encoding: absolute, relative"
echo "  - Normalization: layernorm, rmsnorm"
echo "  - Optimizer: NoamOpt (True), AdamW (False)"
echo "  - Label Smoothing: True, False"
echo ""
echo "================================================================"
echo ""

# ============================================================
# Loop through all combinations
# ============================================================

# Position encodings
for POS_ENC in "absolute" "relative"; do

    # Normalization types
    for NORM_TYPE in "layernorm" "rmsnorm"; do

        # NoamOpt settings
        for USE_NOAMOPT in "True" "False"; do

            # Label smoothing settings
            for USE_SMOOTHING in "True" "False"; do

                # Generate experiment name based on settings
                POS_SUFFIX="abs"
                if [ "$POS_ENC" = "relative" ]; then
                    POS_SUFFIX="rel"
                fi

                NORM_SUFFIX="ln"
                if [ "$NORM_TYPE" = "rmsnorm" ]; then
                    NORM_SUFFIX="rms"
                fi

                NOAM_SUFFIX="noam"
                if [ "$USE_NOAMOPT" = "False" ]; then
                    NOAM_SUFFIX="adam"
                fi

                SMOOTH_SUFFIX="smooth"
                if [ "$USE_SMOOTHING" = "False" ]; then
                    SMOOTH_SUFFIX="nosmooth"
                fi

                EXP_NAME="${DATA_SIZE}_${POS_SUFFIX}_${NORM_SUFFIX}_${NOAM_SUFFIX}_${SMOOTH_SUFFIX}"

                echo "================================================================"
                echo "Starting Experiment: $EXP_NAME"
                echo "================================================================"
                echo "  Position Encoding: $POS_ENC"
                echo "  Normalization: $NORM_TYPE"
                echo "  Use NoamOpt: $USE_NOAMOPT"
                echo "  Use Smoothing: $USE_SMOOTHING"
                echo "================================================================"
                echo ""

                # Build command arguments
                ARGS="--data_size $DATA_SIZE \
                    --batch_size $BATCH_SIZE \
                    --epoch_num $EPOCHS \
                    --position_encoding $POS_ENC \
                    --norm_type $NORM_TYPE \
                    --exp_name $EXP_NAME \
                    --gpu_id $GPU_ID"

                # Add NoamOpt argument
                if [ "$USE_NOAMOPT" = "True" ]; then
                    ARGS="$ARGS --use_noamopt"
                else
                    ARGS="$ARGS --no_noamopt"
                fi

                # Add smoothing argument
                if [ "$USE_SMOOTHING" = "True" ]; then
                    ARGS="$ARGS --use_smoothing"
                else
                    ARGS="$ARGS --no_smoothing"
                fi

                # Add max_relative_position for relative position encoding
                if [ "$POS_ENC" = "relative" ]; then
                    ARGS="$ARGS --max_relative_position 32"
                fi

                # Run the experiment
                python main.py $ARGS

                echo ""
                echo "✓ Experiment $EXP_NAME completed!"
                echo ""

            done # USE_SMOOTHING
        done # USE_NOAMOPT
    done # NORM_TYPE
done # POS_ENC

echo "================================================================"
echo "ALL EXPERIMENTS COMPLETED! (100K Dataset)"
echo "================================================================"
echo ""
echo "Results saved in ./experiment/ directory with subfolders:"
echo "  Format: ${DATA_SIZE}_{pos}_{norm}_{optimizer}_{smoothing}/"
echo ""
echo "Examples:"
echo "  - ./experiment/${DATA_SIZE}_abs_ln_noam_nosmooth/"
echo "  - ./experiment/${DATA_SIZE}_abs_ln_noam_smooth/"
echo "  - ./experiment/${DATA_SIZE}_abs_ln_adam_nosmooth/"
echo "  - ./experiment/${DATA_SIZE}_abs_ln_adam_smooth/"
echo "  - ./experiment/${DATA_SIZE}_rel_ln_noam_nosmooth/"
echo "  - ... (16 total)"
echo ""
echo "Each folder contains:"
echo "  - model_*.pth       (trained model)"
echo "  - train_*.log       (training log)"
echo "  - output_*.txt      (translation outputs)"
echo ""
echo "Next step: Run analyze_results.py to compare performance"
echo "================================================================"
