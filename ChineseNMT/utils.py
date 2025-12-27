import os
import logging
import pickle
import jieba
import sentencepiece as spm
import config  # Import config to get tokenizer path


class JiebaTokenizer:
    """Jieba tokenizer wrapper to match SentencePiece interface"""
    def __init__(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab_info = pickle.load(f)
        self.word2idx = vocab_info['word2idx']
        self.idx2word = vocab_info['idx2word']

    def pad_id(self):
        return self.word2idx.get('<pad>', 0)

    def unk_id(self):
        return self.word2idx.get('<unk>', 1)

    def bos_id(self):
        return self.word2idx.get('<s>', 2)

    def eos_id(self):
        return self.word2idx.get('</s>', 3)

    def EncodeAsIds(self, text):
        """Tokenize Chinese text using Jieba and convert to ids"""
        tokens = list(jieba.cut(text))
        ids = [self.word2idx.get(token, self.unk_id()) for token in tokens]
        return ids

    def DecodeIds(self, ids):
        """Decode ids back to Chinese text"""
        tokens = [self.idx2word.get(id, '<unk>') for id in ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<pad>', '<unk>', '<s>', '</s>']]
        return ''.join(tokens)

    def decode_ids(self, ids):
        """Alias for DecodeIds to match SentencePiece interface"""
        return self.DecodeIds(ids)


def chinese_tokenizer_load():
    """Load Jieba tokenizer with vocabulary - path from config"""
    vocab_path = os.path.join(config.tokenizer_dir, 'chinese_vocab.pkl')
    if os.path.exists(vocab_path):
        return JiebaTokenizer(vocab_path)
    else:
        raise FileNotFoundError(
            f"Chinese vocabulary not found at {vocab_path}.\n"
            f"Current DATA_SIZE: {config.DATA_SIZE}\n"
            f"Please run: python preprocess_data.py {config.DATA_SIZE}"
        )


def english_tokenizer_load():
    """Load SentencePiece tokenizer for English - path from config"""
    model_path = os.path.join(config.tokenizer_dir, 'eng.model')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"English model not found at {model_path}.\n"
            f"Current DATA_SIZE: {config.DATA_SIZE}\n"
            f"Please run: python preprocess_data.py {config.DATA_SIZE}"
        )
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load(model_path)
    return sp_eng


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


