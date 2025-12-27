"""
数据过滤诊断脚本 - 分析为什么99.79%的数据被过滤掉了
"""
import json
import re
from collections import Counter

# 加载数据
DATA_DIR = "/mnt/c/Users/sysu/Desktop/nlp_hw3/dataset_zh_en/AP0004_Midterm&Final_translation_dataset_zh_en"
TRAIN_FILE = f"{DATA_DIR}/train_100k.jsonl"

def clean_text(text, lang='en'):
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    if lang == 'zh':
        text = re.sub(r'\s+', '', text)
    else:
        text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language_ratio(text, lang):
    if lang == 'zh':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        ratio = chinese_chars / len(text) if len(text) > 0 else 0
        return ratio, ratio > 0.7
    else:
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        ratio = english_chars / len(text) if len(text) > 0 else 0
        return ratio, ratio > 0.7

def analyze_sample():
    data = []
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 100:  # 只分析前100条
                data.append(json.loads(line))

    print("="*60)
    print("前100条数据样本分析")
    print("="*60)

    stats = {
        'total': 0,
        'too_short': 0,
        'too_long': 0,
        'ratio_mismatch': 0,
        'language_mismatch': 0,
        'duplicate': 0
    }

    seen_pairs = set()

    for i, item in enumerate(data):
        stats['total'] += 1
        en_text = clean_text(item['en'], lang='en')
        zh_text = clean_text(item['zh'], lang='zh')

        print(f"\n样本 {i+1}:")
        print(f"  原文: zh='{item['zh'][:50]}...'")
        print(f"  原文: en='{item['en'][:50]}...'")
        print(f"  清理后: zh='{zh_text[:50]}...', en='{en_text[:50]}...'")

        # 检查长度
        if len(en_text) < 3 or len(zh_text) < 3:
            stats['too_short'] += 1
            print(f"  ❌ 太短: zh_len={len(zh_text)}, en_len={len(en_text)}")

        if len(en_text) > 100 or len(zh_text) > 100:
            stats['too_long'] += 1
            print(f"  ❌ 太长: zh_len={len(zh_text)}, en_len={len(en_text)}")

        # 检查长度比例
        len_ratio = max(len(en_text), len(zh_text)) / max(min(len(en_text), len(zh_text)), 1)
        if len_ratio > 1.5:
            stats['ratio_mismatch'] += 1
            print(f"  ❌ 长度比例不匹配: {len_ratio:.2f}")

        # 检查语言比例
        en_ratio, en_ok = detect_language_ratio(en_text, 'en')
        zh_ratio, zh_ok = detect_language_ratio(zh_text, 'zh')
        if not en_ok or not zh_ok:
            stats['language_mismatch'] += 1
            print(f"  ❌ 语言检测失败: en_ratio={en_ratio:.2f}({'OK' if en_ok else 'FAIL'}), zh_ratio={zh_ratio:.2f}({'OK' if zh_ok else 'FAIL'})")

        # 检查重复
        pair = (en_text, zh_text)
        if pair in seen_pairs:
            stats['duplicate'] += 1
            print(f"  ❌ 重复数据")
        seen_pairs.add(pair)

        # 如果所有检查都通过
        if (len(en_text) >= 3 and len(zh_text) >= 3 and
            len(en_text) <= 100 and len(zh_text) <= 100 and
            len_ratio <= 1.5 and en_ok and zh_ok):
            print(f"  ✅ 通过所有检查")

    print("\n" + "="*60)
    print("统计结果 (前100条)")
    print("="*60)
    print(f"总数据: {stats['total']}")
    print(f"太短: {stats['too_short']}")
    print(f"太长: {stats['too_long']}")
    print(f"长度比例不匹配: {stats['ratio_mismatch']}")
    print(f"语言检测失败: {stats['language_mismatch']}")
    print(f"重复数据: {stats['duplicate']}")
    print(f"通过检查: {stats['total'] - sum(v for k, v in stats.items() if k != 'total')}")

if __name__ == "__main__":
    analyze_sample()
