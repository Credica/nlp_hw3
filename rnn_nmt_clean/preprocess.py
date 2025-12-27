"""
适配数据集特点的预处理脚本
针对中英文长度差异大的学术/新闻文本优化
"""
import json
import pickle
import os
import re
import argparse
from collections import Counter
import jieba
import sentencepiece as spm
from tqdm import tqdm
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Preprocessing for RNN NMT')
parser.add_argument('--dataset', type=str, default='10k', choices=['10k', '100k'],
                    help='Dataset size to preprocess (10k or 100k)')
args = parser.parse_args()

# 适配数据集特点的过滤参数
MIN_FREQ = 2  # 保持较低阈值，因为学术文本词汇多样
MIN_LEN = 3

# 分别设置中英文长度限制
MAX_LEN_ZH = 150  # 中文更长
MAX_LEN_EN = 400  # 英文更长
MAX_LEN_RATIO = 5.0  # 放宽长度比例限制

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

DATASET_SUFFIX = f"_{args.dataset}"
DATA_DIR = "../dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en"
TRAIN_FILE = f"{DATA_DIR}/train{DATASET_SUFFIX}.jsonl"
VALID_FILE = f"{DATA_DIR}/valid.jsonl"
TEST_FILE = f"{DATA_DIR}/test.jsonl"

TOKENIZER_DIR = f"./tokenizer{DATASET_SUFFIX}"
os.makedirs(TOKENIZER_DIR, exist_ok=True)

def clean_text(text, lang='en'):
    """Clean and normalize text"""
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Handle whitespace based on language
    if lang == 'zh':
        text = re.sub(r'\s+', '', text)
    else:  # English
        text = re.sub(r'\s+', ' ', text)

    return text.strip()

def load_data(file_path):
    """Load JSONL data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def detect_language_ratio(text, lang):
    """检测语言比例，放宽阈值"""
    if lang == 'zh':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        ratio = chinese_chars / len(text) if len(text) > 0 else 0
        return ratio > 0.5  # 从0.7降到0.5
    else:
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        ratio = english_chars / len(text) if len(text) > 0 else 0
        return ratio > 0.5  # 从0.7降到0.5

def analyze_length_distribution(texts, lang):
    """分析长度分布，返回统计信息"""
    lengths = [len(text) for text in texts]
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'percentile_90': np.percentile(lengths, 90),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }

def adaptive_filter_data(data):
    """适配数据集特点的过滤"""
    filtered = []
    stats = {
        'total': 0,
        'too_short': 0,
        'too_long': 0,
        'ratio_mismatch': 0,
        'language_mismatch': 0,
        'duplicate': 0
    }

    seen_pairs = set()

    for item in data:
        stats['total'] += 1
        en_text = clean_text(item['en'], lang='en')
        zh_text = clean_text(item['zh'], lang='zh')

        # 检查重复
        pair = (en_text, zh_text)
        if pair in seen_pairs:
            stats['duplicate'] += 1
            continue
        seen_pairs.add(pair)

        # 检查长度（分别设置中英文限制）
        if len(en_text) < MIN_LEN or len(zh_text) < MIN_LEN:
            stats['too_short'] += 1
            continue

        if len(en_text) > MAX_LEN_EN or len(zh_text) > MAX_LEN_ZH:
            stats['too_long'] += 1
            continue

        # 检查中英文长度比例（放宽阈值）
        len_ratio = max(len(en_text), len(zh_text)) / max(min(len(en_text), len(zh_text)), 1)
        if len_ratio > MAX_LEN_RATIO:
            stats['ratio_mismatch'] += 1
            continue

        # 检查语言比例（放宽阈值）
        if not detect_language_ratio(en_text, 'en') or not detect_language_ratio(zh_text, 'zh'):
            stats['language_mismatch'] += 1
            continue

        filtered.append({
            'en': en_text,
            'zh': zh_text
        })

    return filtered, stats

def tokenize_chinese(texts):
    """Tokenize Chinese texts with Jieba"""
    print("Tokenizing Chinese with Jieba...")

    word_freq = Counter()
    for text in tqdm(texts, desc="Tokenizing Chinese"):
        words = list(jieba.cut(text))
        word_freq.update(words)

    # 过滤低频词
    vocab = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    for word, freq in word_freq.most_common():
        if freq >= MIN_FREQ:
            vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    print(f"Chinese vocabulary size: {len(vocab)}")
    print(f"Total unique words: {len(word_freq)}")

    return vocab, word2idx, idx2word, word_freq

def tokenize_english(texts):
    """Tokenize English texts with SentencePiece BPE"""
    print("Tokenizing English with SentencePiece BPE...")

    # Save to temporary file
    temp_file = "./temp_en.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    # Train BPE model
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=f'{TOKENIZER_DIR}/eng',
        vocab_size=8000,
        character_coverage=0.9995,
        model_type='bpe',
        bos_id=2,
        eos_id=3,
        pad_id=0,
        unk_id=1,
    )

    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.load(f'{TOKENIZER_DIR}/eng.model')

    vocab_size = sp.get_piece_size()
    print(f"English vocabulary size: {vocab_size}")

    os.remove(temp_file)
    return sp

def encode_data(data, zh_word2idx, en_sp):
    """Encode data to numerical format"""
    encoded_data = []

    for item in tqdm(data, desc="Encoding data"):
        zh_words = list(jieba.cut(item['zh']))
        zh_ids = [zh_word2idx.get(word, zh_word2idx[UNK_TOKEN])
                 for word in zh_words]
        zh_ids = [zh_word2idx[SOS_TOKEN]] + zh_ids + [zh_word2idx[EOS_TOKEN]]

        en_ids = en_sp.encode_as_ids(item['en'])
        en_ids = [en_sp.bos_id()] + en_ids + [en_sp.eos_id()]

        encoded_data.append({
            'zh': zh_ids,
            'en': en_ids
        })

    return encoded_data

def save_analysis_report(zh_stats, en_stats, filter_stats, vocab_stats):
    """保存数据分析报告"""
    report = {
        'filtering_stats': filter_stats,
        'zh_length_stats': zh_stats,
        'en_length_stats': en_stats,
        'vocab_stats': {
            'zh_size': len(vocab_stats['vocab']),
            'zh_unique_words': len(vocab_stats['word_freq']),
        }
    }

    with open(f'{TOKENIZER_DIR}/analysis_report.pkl', 'wb') as f:
        pickle.dump(report, f)

    print("\n" + "="*60)
    print("数据质量分析报告")
    print("="*60)
    print(f"原始数据: {filter_stats['total']} 条")
    kept = filter_stats['total'] - sum([v for k, v in filter_stats.items() if k != 'total'])
    print(f"过滤后数据: {kept} 条")
    print(f"保留率: {100 * kept / filter_stats['total']:.2f}%")
    print("\n过滤原因:")
    print(f"  - 太短: {filter_stats['too_short']}")
    print(f"  - 太长 (zh>{MAX_LEN_ZH}, en>{MAX_LEN_EN}): {filter_stats['too_long']}")
    print(f"  - 长度比例不匹配 (>1:{MAX_LEN_RATIO}): {filter_stats['ratio_mismatch']}")
    print(f"  - 语言检测失败: {filter_stats['language_mismatch']}")
    print(f"  - 重复数据: {filter_stats['duplicate']}")
    print(f"\n中文长度统计 (字符):")
    print(f"  - 平均: {zh_stats['mean']:.1f}, 中位数: {zh_stats['median']:.1f}")
    print(f"  - 90%: {zh_stats['percentile_90']:.1f}, 95%: {zh_stats['percentile_95']:.1f}")
    print(f"  - 最大: {zh_stats['max']}")
    print(f"\n英文长度统计 (字符):")
    print(f"  - 平均: {en_stats['mean']:.1f}, 中位数: {en_stats['median']:.1f}")
    print(f"  - 90%: {en_stats['percentile_90']:.1f}, 95%: {en_stats['percentile_95']:.1f}")
    print(f"  - 最大: {en_stats['max']}")

def main():
    print("=" * 60)
    print("适配数据集特点的预处理")
    print("=" * 60)
    print(f"参数: zh_len≤{MAX_LEN_ZH}, en_len≤{MAX_LEN_EN}, 比例≤1:{MAX_LEN_RATIO}")

    # Load data
    print("\n1. 加载数据...")
    train_data = load_data(TRAIN_FILE)
    valid_data = load_data(VALID_FILE)
    test_data = load_data(TEST_FILE)

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(valid_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # Adaptive filtering
    print("\n2. 适配性过滤...")
    train_filtered, filter_stats = adaptive_filter_data(train_data)
    valid_filtered, _ = adaptive_filter_data(valid_data)
    test_filtered, _ = adaptive_filter_data(test_data)

    print(f"过滤后 - 训练: {len(train_filtered)}, 验证: {len(valid_filtered)}, 测试: {len(test_filtered)}")

    # Analyze length distribution
    print("\n3. 分析长度分布...")
    zh_texts = [item['zh'] for item in train_filtered]
    en_texts = [item['en'] for item in train_filtered]

    zh_stats = analyze_length_distribution(zh_texts, 'zh')
    en_stats = analyze_length_distribution(en_texts, 'en')

    # Tokenize and build vocab
    print("\n4. 构建中文词汇表...")
    zh_vocab, zh_word2idx, zh_idx2word, zh_word_freq = tokenize_chinese(zh_texts)

    with open(f'{TOKENIZER_DIR}/zh_vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab': zh_vocab,
            'word2idx': zh_word2idx,
            'idx2word': zh_idx2word
        }, f)

    print("\n5. 训练英文BPE...")
    en_sp = tokenize_english(en_texts)

    # Encode data
    print("\n6. 编码数据...")
    train_encoded = encode_data(train_filtered, zh_word2idx, en_sp)
    valid_encoded = encode_data(valid_filtered, zh_word2idx, en_sp)
    test_encoded = encode_data(test_filtered, zh_word2idx, en_sp)

    # Save encoded data
    print("\n7. 保存预处理数据...")

    with open(f'{TOKENIZER_DIR}/train_data.pkl', 'wb') as f:
        pickle.dump(train_encoded, f)

    with open(f'{TOKENIZER_DIR}/valid_data.pkl', 'wb') as f:
        pickle.dump(valid_encoded, f)

    with open(f'{TOKENIZER_DIR}/test_data.pkl', 'wb') as f:
        pickle.dump(test_encoded, f)

    # Save configuration info
    config_info = {
        'zh_vocab_size': len(zh_vocab),
        'en_vocab_size': en_sp.get_piece_size(),
        'train_size': len(train_encoded),
        'valid_size': len(valid_encoded),
        'test_size': len(test_encoded)
    }

    with open(f'{TOKENIZER_DIR}/config_info.pkl', 'wb') as f:
        pickle.dump(config_info, f)

    # Save analysis report
    print("\n8. 保存分析报告...")
    save_analysis_report(zh_stats, en_stats, filter_stats, {
        'vocab': zh_vocab,
        'word_freq': zh_word_freq
    })

    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)
    print(f"中文词汇表大小: {len(zh_vocab)}")
    print(f"英文词汇表大小: {en_sp.get_piece_size()}")
    print(f"训练样本: {len(train_encoded)}")
    print(f"验证样本: {len(valid_encoded)}")
    print(f"测试样本: {len(test_encoded)}")
    print(f"\n词表和分词器保存到: {TOKENIZER_DIR}/")
    print(f"数据保存到: {TOKENIZER_DIR}/")

    # Show examples
    print("\n" + "=" * 60)
    print("样本数据:")
    print("=" * 60)
    sample = train_encoded[0]
    print(f"中文ID: {sample['zh'][:20]}...")
    print(f"英文ID: {sample['en'][:20]}...")

    zh_text = ' '.join([zh_idx2word[idx] for idx in sample['zh']])
    en_text = en_sp.decode_ids(sample['en'])
    print(f"\n中文(解码): {zh_text}")
    print(f"英文(解码): {en_text}")

if __name__ == "__main__":
    main()
