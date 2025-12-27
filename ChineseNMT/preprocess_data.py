#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 中文→英文翻译
从 JSONL 格式转换为 ChineseNMT 所需格式
使用 Jieba 处理中文，SentencePiece 处理英文

Usage:
    python preprocess_data.py
"""

import json
import os
import re
import pickle
from collections import Counter
from tqdm import tqdm
import jieba
import sentencepiece as spm
import config  # Import config module


class DataPreprocessor:
    """数据预处理器 - 中文→英文"""

    def __init__(self,
                 source_data_dir="/mnt/c/Users/sysu/Desktop/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en",
                 train_size="10k"):
        """
        Args:
            source_data_dir: 源数据目录
            train_size: 训练集大小，可选 "10k" 或 "100k"
        """
        self.source_data_dir = source_data_dir
        self.train_size = train_size

        # 根据数据集大小自动设置输出目录
        self.output_dir = f"./data_{train_size}"
        self.tokenizer_dir = f"./data_{train_size}/tokenizer"

        self.makedirs()

    def makedirs(self):
        """创建必要的目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'json'), exist_ok=True)
        os.makedirs(self.tokenizer_dir, exist_ok=True)

    def clean_text(self, text, lang='en'):
        """清理文本，移除非法字符"""
        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # 清理多余的空白字符
        if lang == 'en':
            text = re.sub(r'\s+', ' ', text)
        else:  # 中文
            text = re.sub(r'\s+', '', text)

        return text.strip()

    def load_jsonl(self, file_path):
        """加载JSONL文件"""
        data = []
        print(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def clean_dataset(self, data, max_len=200, min_len=3):
        """清理整个数据集"""
        cleaned = []
        for item in tqdm(data, desc="Cleaning dataset"):
            zh = self.clean_text(item['zh'], 'zh')
            en = self.clean_text(item['en'], 'en')

            # 检查长度限制
            if min_len <= len(zh) <= max_len and min_len <= len(en) <= max_len:
                # 注意：[中文, 英文] - 中文作为源语言，英文作为目标语言
                cleaned.append([zh, en])

        return cleaned

    def save_json(self, data, file_path):
        """保存为JSON格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved to {file_path}")

    def build_chinese_vocab_with_jieba(self, texts, min_freq=2):
        """使用Jieba构建中文词汇表"""
        print("Building Chinese vocabulary with Jieba...")

        # 收集所有中文词汇
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing Chinese"):
            tokens = list(jieba.cut(text))
            all_tokens.extend(tokens)

        # 统计词频
        word_freq = Counter(all_tokens)

        # 构建词汇表（包含特殊符号）
        special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        vocab = special_tokens.copy()

        # 添加高频词
        for word, freq in word_freq.most_common():
            if freq >= min_freq and word not in vocab and word.strip():
                vocab.append(word)

        # 创建词汇映射
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}

        print(f"Chinese vocabulary size: {len(vocab)}")
        print(f"Min frequency: {min_freq}")

        # 保存词汇表
        vocab_info = {
            'word2idx': word2idx,
            'idx2word': idx2word,
            'word_freq': dict(word_freq),
            'min_freq': min_freq,
            'vocab_size': len(vocab)
        }

        vocab_path = os.path.join(self.tokenizer_dir, 'chinese_vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_info, f)

        print(f"Chinese vocabulary saved to {vocab_path}")
        return vocab_info

    def train_english_sentencepiece(self, texts, vocab_size=8000):
        """为英文训练SentencePiece BPE模型"""
        print("\nTraining English SentencePiece model...")

        # 准备训练文本
        temp_file = os.path.join(self.tokenizer_dir, "eng_corpus.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        print(f"Training corpus size: {len(texts)} sentences")
        print(f"Vocabulary size: {vocab_size}")

        model_prefix = os.path.join(self.tokenizer_dir, 'eng')

        # 训练BPE模型
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            model_type='bpe',
            vocab_size=vocab_size,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            split_digits=True
        )

        # 清理临时文件
        os.remove(temp_file)

        print(f"English model saved: {model_prefix}.model")
        print(f"English vocab saved: {model_prefix}.vocab")

    def process_all_data(self):
        """处理所有数据集"""
        print("="*60)
        print("Data Preprocessing for Chinese→English NMT")
        print("="*60)
        print(f"Training data size: {self.train_size}")
        print("="*60)

        # 文件路径
        train_path = os.path.join(self.source_data_dir, f"train_{self.train_size}.jsonl")
        valid_path = os.path.join(self.source_data_dir, "valid.jsonl")
        test_path = os.path.join(self.source_data_dir, "test.jsonl")

        # 检查文件是否存在
        for path in [train_path, valid_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

        # 加载数据
        print("\n[1/4] Loading data...")
        train_data = self.load_jsonl(train_path)
        valid_data = self.load_jsonl(valid_path)
        test_data = self.load_jsonl(test_path)

        print(f"Train: {len(train_data)} samples")
        print(f"Valid: {len(valid_data)} samples")
        print(f"Test: {len(test_data)} samples")

        # 清理数据
        print("\n[2/4] Cleaning data...")
        train_clean = self.clean_dataset(train_data)
        valid_clean = self.clean_dataset(valid_data)
        test_clean = self.clean_dataset(test_data)

        print(f"After cleaning - Train: {len(train_clean)}, Valid: {len(valid_clean)}, Test: {len(test_clean)}")

        # 保存清理后的数据
        print("\n[3/4] Saving cleaned data...")
        self.save_json(train_clean, os.path.join(self.output_dir, 'json', 'train.json'))
        self.save_json(valid_clean, os.path.join(self.output_dir, 'json', 'dev.json'))
        self.save_json(test_clean, os.path.join(self.output_dir, 'json', 'test.json'))

        # 提取中文和英文文本
        zh_texts = [item[0] for item in train_clean]  # 中文（源语言）
        en_texts = [item[1] for item in train_clean]  # 英文（目标语言）

        # 训练分词器
        print("\n[4/4] Training tokenizers...")

        # 中文：使用Jieba + 词汇表
        zh_vocab_info = self.build_chinese_vocab_with_jieba(zh_texts, min_freq=config.min_freq)

        # 英文：使用SentencePiece BPE
        self.train_english_sentencepiece(en_texts, vocab_size=8000)

        print("\n" + "="*60)
        print("Preprocessing completed successfully!")
        print("="*60)
        print(f"Data directory: {self.output_dir}")
        print(f"Tokenizer directory: {self.tokenizer_dir}")
        print(f"\nChinese (source) vocab size: {zh_vocab_info['vocab_size']}")
        print(f"English (target) vocab size: 8000")
        print(f"\n✅ Dataset '{self.train_size}' is ready!")
        print(f"   - Update config.py: set DATA_SIZE = '{self.train_size}'")
        print(f"   - Update config.py: set src_vocab_size = {zh_vocab_info['vocab_size']}")
        print(f"\nYou can now train the model using: python main.py")
        print("="*60)


def main():
    """主函数"""
    import sys

    # 支持命令行参数选择数据集大小
    if len(sys.argv) > 1:
        train_size = sys.argv[1]
        if train_size not in ["10k", "100k"]:
            print("Error: train_size must be '10k' or '100k'")
            print("Usage: python preprocess_data.py [10k|100k]")
            sys.exit(1)
    else:
        # 默认处理两个数据集
        print("="*60)
        print("Processing BOTH datasets (10k and 100k)")
        print("="*60)
        print("You can also specify one dataset:")
        print("  python preprocess_data.py 10k")
        print("  python preprocess_data.py 100k")
        print("="*60)
        print()

        # 处理 10k 数据集
        print("📦 [1/2] Processing 10k dataset...")
        preprocessor_10k = DataPreprocessor(
            source_data_dir="/mnt/c/Users/sysu/Desktop/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en",
            train_size="10k"
        )
        preprocessor_10k.process_all_data()

        print("\n" + "="*60)
        print()

        # 处理 100k 数据集
        print("📦 [2/2] Processing 100k dataset...")
        preprocessor_100k = DataPreprocessor(
            source_data_dir="/mnt/c/Users/sysu/Desktop/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en",
            train_size="100k"
        )
        preprocessor_100k.process_all_data()

        print("\n" + "="*80)
        print("🎉 All datasets preprocessed successfully!")
        print("="*80)
        print("\nNext steps:")
        print("1. Choose dataset in config.py: set DATA_SIZE = '10k' or '100k'")
        print("2. Update src_vocab_size in config.py based on output above")
        print("3. Run training: python main.py")
        print("="*80)
        return

    # 处理单个数据集
    preprocessor = DataPreprocessor(
        source_data_dir="/mnt/c/Users/sysu/Desktop/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en",
        train_size=train_size
    )
    preprocessor.process_all_data()


if __name__ == "__main__":
    main()
