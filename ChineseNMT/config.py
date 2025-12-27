import torch

# ============================================================
# Dataset Selection
# ============================================================
# Choose which dataset to use: "10k" or "100k"
# Remember to update src_vocab_size after preprocessing!
DATA_SIZE = "100k"  # Change to "100k" for larger dataset

# ============================================================
# Data Preprocessing Parameters
# ============================================================
# Min frequency for vocabulary filtering (in Jieba)
min_freq = 2  # Filter out words appearing less than 2 times (降低min_freq增加vocab覆盖)

# ============================================================
# Architecture Ablation Parameters
# ============================================================
# Position encoding type: 'absolute' or 'relative'
position_encoding_type = 'absolute'  # Options: 'absolute', 'relative'

# Normalization type: 'layernorm' or 'rmsnorm'
norm_type = 'layernorm'  # Options: 'layernorm', 'rmsnorm'

# Relative position encoding parameters (only used when position_encoding_type='relative')
max_relative_position = 32  # Maximum relative position distance

# ============================================================
# Model Architecture Parameters
# ============================================================
d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3

# Vocabulary sizes
# Update these values after running preprocess_data.py for each dataset
# 10k dataset vocab sizes:
src_vocab_size_10k = 7697   # Chinese vocabulary (Jieba) for 10k
# 100k dataset vocab sizes:
src_vocab_size_100k = 14769  # Chinese vocabulary (Jieba) for 100k （min_freq=3->27229）

# Automatically select vocab size based on DATA_SIZE
src_vocab_size = src_vocab_size_10k if DATA_SIZE == "10k" else src_vocab_size_100k
tgt_vocab_size = 8000   # English vocabulary (SentencePiece BPE) - same for both

# Training parameters
batch_size = 64         # Batch size (adjust based on GPU memory)
epoch_num = 30          # Training epochs
early_stop = 5          # Early stopping patience
# Fixed learning rate for stable training
lr = 0.0005             # Used when use_noamopt=False

# Decoding parameters
max_len = 80            # Maximum decoding length (increased for safety)
beam_size = 5           # Beam size for beam search
# Label Smoothing - DISABLED for small dataset (10k)
# Label smoothing with 0.1 may be too aggressive for small datasets
# Will cause training instability
use_smoothing = False  
# NoamOpt - DISABLED for stable training on small dataset
use_noamopt = True  

# Data paths - automatically set based on DATA_SIZE
data_dir = f'./data_{DATA_SIZE}'
train_data_path = f'./data_{DATA_SIZE}/json/train.json'
dev_data_path = f'./data_{DATA_SIZE}/json/dev.json'
test_data_path = f'./data_{DATA_SIZE}/json/test.json'
tokenizer_dir = f'./data_{DATA_SIZE}/tokenizer'

# Model and output paths
model_path = f'./experiment/model_{DATA_SIZE}.pth'
log_path = f'./experiment/train_{DATA_SIZE}.log'
output_path = f'./experiment/output_{DATA_SIZE}.txt'

# GPU configuration for single GPU
# gpu_id is the actual GPU device ID
# device_id is for DataParallel (use [0] for single GPU)
gpu_id = '0'
device_id = [0]  # Single GPU: only use device 0

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
