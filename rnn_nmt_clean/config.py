"""
Configuration for RNN-based Neural Machine Translation
"""
import torch
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data configuration
DATA_DIR = "/mnt/c/Users/sysu/Desktop/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en"
TRAIN_FILE = f"{DATA_DIR}/train_10k.jsonl"
VALID_FILE = f"{DATA_DIR}/valid.jsonl"
TEST_FILE = f"{DATA_DIR}/test.jsonl"

# Tokenizer configuration
TOKENIZER_DIR = "./tokenizer"
MIN_FREQ = 2  # Minimum frequency for vocabulary

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
BATCH_SIZE = 32
N_EPOCHS = 30
CLIP = 1
LEARNING_RATE = 0.0005
TEACHER_FORCING_RATIO = 0.5  # Based on your experiments - this is the best!
EARLY_STOPPING_PATIENCE = 5

# Data filtering
MAX_LEN = 100  # Maximum sentence length
MIN_LEN = 3   # Minimum sentence length

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# Model saving
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = f"{SAVE_DIR}/rnn_nmt_best.pt"
LOG_PATH = f"{SAVE_DIR}/train.log"
RESULTS_PATH = f"{SAVE_DIR}/results.txt"
