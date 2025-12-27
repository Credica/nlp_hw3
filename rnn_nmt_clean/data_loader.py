import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle

# Import config_v2 for dataset-specific paths
import config_v2 as config

class MTDataset(Dataset):
    """Dataset for Machine Translation"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'src': torch.tensor(item['zh'], dtype=torch.long),  # Chinese (source)
            'trg': torch.tensor(item['en'], dtype=torch.long),  # English (target)
        }

def collate_fn(batch):
    """Collate function for data loader"""
    # Extract source and target sequences
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]

    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=False, padding_value=0)
    trg_padded = pad_sequence(trg_batch, batch_first=False, padding_value=0)

    return {
        'src': src_padded,
        'trg': trg_padded,
    }

def get_data_loaders(config_obj=None):
    """Get training and validation data loaders"""
    if config_obj is None:
        config_obj = config

    # Load preprocessed data
    print("Loading preprocessed data...")
    with open(f'{config_obj.TOKENIZER_DIR}/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(f'{config_obj.TOKENIZER_DIR}/valid_data.pkl', 'rb') as f:
        valid_data = pickle.load(f)

    with open(f'{config_obj.TOKENIZER_DIR}/config_info.pkl', 'rb') as f:
        config_info = pickle.load(f)

    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(valid_data)} validation samples")

    # Update config with vocabulary sizes
    config_obj.INPUT_DIM = config_info['zh_vocab_size']
    config_obj.OUTPUT_DIM = config_info['en_vocab_size']

    # Create datasets
    train_dataset = MTDataset(train_data)
    valid_dataset = MTDataset(valid_data)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_obj.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config_obj.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, valid_loader

def get_test_loader(config_obj=None):
    """Get test data loader"""
    if config_obj is None:
        config_obj = config

    # Load test data
    with open(f'{config_obj.TOKENIZER_DIR}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print(f"Loaded {len(test_data)} test samples")

    # Create dataset
    test_dataset = MTDataset(test_data)

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config_obj.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader
