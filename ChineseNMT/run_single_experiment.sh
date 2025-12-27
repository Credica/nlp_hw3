#!/bin/bash

# ============================================================
# Single Experiment Runner
# ============================================================
# This script runs a single experiment with custom parameters.
# Usage examples:
#
# 1. Test Relative Position Encoding:
#    bash run_single_experiment.sh --position_encoding relative
#
# 2. Test RMSNorm:
#    bash run_single_experiment.sh --norm_type rmsnorm
#
# 3. Combined:
#    bash run_single_experiment.sh --position_encoding relative --norm_type rmsnorm
#
# 4. With custom model size:
#    bash run_single_experiment.sh --d_model 256 --n_heads 4 --n_layers 4
# ============================================================

# Default parameters (can be overridden by command line arguments)
DATA_SIZE="10k"
BATCH_SIZE=32
EPOCHS=40
GPU_ID=0

# Show usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: bash run_single_experiment.sh [OPTIONS]"
    echo ""
    echo "Architecture Ablation Options:"
    echo "  --position_encoding TYPE    Position encoding type: absolute or relative (default: absolute)"
    echo "  --norm_type TYPE           Normalization type: layernorm or rmsnorm (default: layernorm)"
    echo "  --max_relative_position N  Maximum relative position (default: 32)"
    echo ""
    echo "Model Architecture Options:"
    echo "  --d_model N                Model dimension (default: 512)"
    echo "  --n_heads N                Number of attention heads (default: 8)"
    echo "  --n_layers N               Number of layers (default: 6)"
    echo "  --d_ff N                   Feed-forward dimension (default: 2048)"
    echo "  --dropout RATE             Dropout rate (default: 0.1)"
    echo ""
    echo "Training Options:"
    echo "  --data_size SIZE           Dataset size: 10k or 100k (default: 10k)"
    echo "  --batch_size N             Batch size (default: 32)"
    echo "  --epoch_num N              Number of epochs (default: 40)"
    echo "  --lr RATE                  Learning rate (default: 1e-4)"
    echo "  --no_noamopt               Don't use Noam optimizer"
    echo "  --no_smoothing             Don't use label smoothing"
    echo ""
    echo "Path and Naming Options:"
    echo "  --exp_name NAME            Experiment name (files saved in ./experiment/)"
    echo "  --model_path PATH          Custom model save path"
    echo "  --log_path PATH            Custom log save path"
    echo "  --output_path PATH         Custom output save path"
    echo "  --gpu_id ID                GPU device ID (default: 0)"
    echo ""
    echo "Examples:"
    echo "  bash run_single_experiment.sh --position_encoding relative"
    echo "  bash run_single_experiment.sh --norm_type rmsnorm"
    echo "  bash run_single_experiment.sh --exp_name \"my_exp\""
    echo "  bash run_single_experiment.sh --model_path \"./models/model.pth\""
    exit 0
fi

# Run the experiment with all provided arguments
echo "Starting Transformer NMT Training..."
echo "Parameters: $@"
echo ""

python main.py "$@"

echo ""
echo "Experiment completed!"
