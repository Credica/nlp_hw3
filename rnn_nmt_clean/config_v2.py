"""
Configuration for RNN-based Neural Machine Translation
Updated to match assignment requirements:
- Unidirectional encoder/decoder
- Multiple attention mechanisms
- Multiple training strategies
- Multiple decoding strategies
"""
import torch
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='RNN NMT Configuration')
parser.add_argument('--dataset', type=str, default='10k', choices=['10k', '100k'],
                    help='Dataset size to use (10k or 100k)')
# args = parser.parse_args()

args, _ = parser.parse_known_args()


# Dataset size suffix
# dataset '10k' -> suffix '_10k', dataset '100k' -> suffix '_100k'
DATASET_SUFFIX = f"_{args.dataset}"

# Random seed for reproducibility
SEED = 42

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data configuration
DATA_DIR = "/mnt/afs/250010024/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en"
TRAIN_FILE = f"{DATA_DIR}/train{DATASET_SUFFIX}.jsonl"
VALID_FILE = f"{DATA_DIR}/valid.jsonl"
TEST_FILE = f"{DATA_DIR}/test.jsonl"

# Tokenizer configuration
TOKENIZER_DIR = f"./tokenizer{DATASET_SUFFIX}"
MIN_FREQ = 5  # Minimum frequency for vocabulary

# Model configuration
INPUT_DIM = None  # Will be set after building vocabulary
OUTPUT_DIM = None  # Will be set after building vocabulary
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Training configuration
BATCH_SIZE = 128
N_EPOCHS = 50
CLIP = 1
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 5

# Training strategies to compare
TEACHER_FORCING_RATIOS = [1.0, 0.5, 0.0]  # 100%, 50%, 0% (Free Running)

# Attention mechanisms to compare
ATTENTION_TYPES = ['dot', 'multiplicative', 'additive']

# Decoding strategies to compare
BEAM_SIZES = [1, 3, 5, 7,10]  # 1=greedy, 3, 5, 10

# Data filtering
MAX_LEN = 100  # Maximum sentence length
MIN_LEN = 3   # Minimum sentence length

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# Model saving
SAVE_DIR = f"./checkpoints{DATASET_SUFFIX}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Default model path
MODEL_PATH = f"{SAVE_DIR}/model_best.pt"

# Experiment results
EXPERIMENT_RESULTS_PATH = f"{SAVE_DIR}/experiment_results.txt"

# Individual experiment result paths
RESULTS_PATH = f"{SAVE_DIR}/results.txt"
