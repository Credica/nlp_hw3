#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer.py - RNN NMT 推理脚本 (Chinese -> English)

依赖你的工程文件：
- config_v2.py
- model_v2.py (build_model)
- beam_search_correct.py (greedy_decode / beam_search_decode)
- preprocess.py 里同款：中文 jieba 分词 + zh_vocab.pkl；英文 sentencepiece (eng.model)

用法示例：
1) 单句：
   python infer.py --dataset 10k --text "随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件。"

2) 文件（每行一句中文）：
   python infer.py --dataset 10k --input zh.txt --output pred.txt --decode beam --beam_size 5

3) 交互：
   python infer.py --dataset 10k --interactive

4) 直接跑预处理后的 test_data.pkl：
   python infer.py --dataset 10k --test_pkl --output test_pred.txt
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import jieba
import pickle
import sentencepiece as spm

# ------------------------------------------------------------
# 关键：先从命令行“偷看”dataset，再 import config
# 避免 config 自己 argparse 把我们其它参数吃掉/报错
# (同 train.py 的处理思路)
# ------------------------------------------------------------
def _peek_dataset(default: str = "10k") -> str:
    dataset = default
    if "--dataset" in sys.argv:
        i = sys.argv.index("--dataset")
        if i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
    return dataset


DATASET = _peek_dataset(default="10k")
_original_argv = sys.argv.copy()
sys.argv = ["infer.py", "--dataset", DATASET]  # 只给 config 它认识的参数

import config_v2 as config  # noqa: E402

sys.argv = _original_argv  # 恢复完整参数给我们自己的 argparse

from model_v2 import build_model  # noqa: E402
from beam_search_correct import greedy_decode, beam_search_decode  # noqa: E402


# -----------------------------
# 参数
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RNN NMT Inference (v2)")
    p.add_argument("--dataset", type=str, default=DATASET, choices=["10k", "100k"], help="dataset size")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: config.MODEL_PATH)")
    p.add_argument("--attention", type=str, default="additive",
                   choices=["additive", "dot", "multiplicative"], help="attention type for model build")

    p.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"], help="decoding strategy")
    p.add_argument("--beam_size", type=int, default=5, help="beam size (only used when --decode beam)")
    p.add_argument("--max_len", type=int, default=100, help="max decoding length")
    p.add_argument("--length_penalty", type=float, default=0.6, help="beam length penalty alpha")

    p.add_argument("--cpu", action="store_true", help="force cpu")
    p.add_argument("--text", type=str, default=None, help="translate a single Chinese sentence")
    p.add_argument("--input", type=str, default=None, help="input file, one Chinese sentence per line")
    p.add_argument("--output", type=str, default=None, help="output file path")
    p.add_argument("--interactive", action="store_true", help="interactive mode")
    p.add_argument("--test_pkl", action="store_true", help="translate TOKENIZER_DIR/test_data.pkl")
    return p.parse_args()


# -----------------------------
# 资源加载：词表 & spm
# -----------------------------
def load_zh_vocab(tokenizer_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab_path = os.path.join(tokenizer_dir, "zh_vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"找不到 {vocab_path}\n"
            f"请先运行预处理：python preprocess.py --dataset {DATASET}"
        )

    with open(vocab_path, "rb") as f:
        zh_data = pickle.load(f)

    return zh_data["word2idx"], zh_data["idx2word"]


def load_en_sp(tokenizer_dir: str) -> spm.SentencePieceProcessor:
    model_path = os.path.join(tokenizer_dir, "eng.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到 {model_path}\n"
            f"请先运行预处理：python preprocess.py --dataset {DATASET}"
        )
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def load_config_info_and_update_dims(tokenizer_dir: str) -> None:
    """
    data_loader / train.py 的做法：读取 config_info.pkl 来设置 INPUT_DIM / OUTPUT_DIM
    """
    info_path = os.path.join(tokenizer_dir, "config_info.pkl")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"找不到 {info_path}\n"
            f"请先运行预处理：python preprocess.py --dataset {DATASET}"
        )
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    config.INPUT_DIM = info["zh_vocab_size"]
    config.OUTPUT_DIM = info["en_vocab_size"]


# -----------------------------
# 中文编码（与 preprocess.py encode_data 同逻辑）
# -----------------------------
def encode_zh(text: str, zh_word2idx: Dict[str, int]) -> torch.Tensor:
    PAD = zh_word2idx.get("<pad>", 0)
    UNK = zh_word2idx.get("<unk>", 1)
    SOS = zh_word2idx.get("<sos>", 2)
    EOS = zh_word2idx.get("<eos>", 3)

    words = list(jieba.cut(text.strip()))
    ids = [zh_word2idx.get(w, UNK) for w in words]
    ids = [SOS] + ids + [EOS]

    # 返回 shape: [src_len]
    return torch.tensor(ids, dtype=torch.long)


# -----------------------------
# 翻译单句（复用 beam_search_correct 的 decode）
# -----------------------------
def translate_one(
    model: torch.nn.Module,
    src_ids: torch.Tensor,
    en_sp: spm.SentencePieceProcessor,
    decode: str,
    beam_size: int,
    max_len: int,
    length_penalty: float,
) -> str:
    # beam_search_correct 内部会用 config.device
    if decode == "greedy":
        pred_ids = greedy_decode(model, src_ids, max_len=max_len)
    else:
        pred_ids = beam_search_decode(
            model,
            src_ids,
            beam_size=beam_size,
            max_len=max_len,
            length_penalty=length_penalty
        )

    # 你的 decode 函数已经去掉了 PAD/SOS/EOS，所以这里直接 decode
    text = en_sp.decode_ids(pred_ids).strip()
    return text


# -----------------------------
# 主流程
# -----------------------------
def main():
    args = parse_args()

    # 覆盖设备
    if args.cpu:
        config.device = torch.device("cpu")
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 覆盖一些推理超参（beam_search_correct 会读取 config.device）
    ckpt_path = args.ckpt or config.MODEL_PATH
    tokenizer_dir = config.TOKENIZER_DIR

    # 读取 vocab size（必须先做，否则 build_model 时 INPUT_DIM/OUTPUT_DIM 还是 None）
    load_config_info_and_update_dims(tokenizer_dir)

    # 加载词表/分词器
    zh_word2idx, _ = load_zh_vocab(tokenizer_dir)
    en_sp = load_en_sp(tokenizer_dir)

    # 构建模型并加载权重
    model = build_model(attention_type=args.attention)
    model.to(config.device)
    model.eval()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到 checkpoint：{ckpt_path}\n"
            f"你可以用 --ckpt 指定，或确认训练保存到 config.MODEL_PATH"
        )

    state = torch.load(ckpt_path, map_location=config.device)
    # 训练脚本保存的是 state_dict
    model.load_state_dict(state, strict=True)

    # 1) 单句
    if args.text is not None:
        src_ids = encode_zh(args.text, zh_word2idx).to(config.device)
        out = translate_one(
            model, src_ids, en_sp,
            decode=args.decode,
            beam_size=args.beam_size,
            max_len=args.max_len,
            length_penalty=args.length_penalty
        )
        print(out)
        return

    # 2) test_data.pkl 整集推理
    if args.test_pkl:
        test_pkl_path = os.path.join(tokenizer_dir, "test_data.pkl")
        if not os.path.exists(test_pkl_path):
            raise FileNotFoundError(f"找不到 {test_pkl_path}")

        with open(test_pkl_path, "rb") as f:
            test_data = pickle.load(f)  # list[{'zh': [...], 'en': [...]}]

        preds: List[str] = []
        for item in test_data:
            src_ids = torch.tensor(item["zh"], dtype=torch.long).to(config.device)
            pred = translate_one(
                model, src_ids, en_sp,
                decode=args.decode,
                beam_size=args.beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty
            )
            preds.append(pred)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as w:
                for p in preds:
                    w.write(p + "\n")
            print(f"[OK] saved: {args.output}")
        else:
            for p in preds[:20]:
                print(p)
            print(f"... (total {len(preds)})")
        return

    # 3) 文件逐行
    if args.input is not None:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"找不到输入文件：{args.input}")

        with open(args.input, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        lines = [ln for ln in lines if ln]

        preds: List[str] = []
        for line in lines:
            src_ids = encode_zh(line, zh_word2idx).to(config.device)
            pred = translate_one(
                model, src_ids, en_sp,
                decode=args.decode,
                beam_size=args.beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty
            )
            preds.append(pred)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as w:
                for p in preds:
                    w.write(p + "\n")
            print(f"[OK] saved: {args.output}")
        else:
            for p in preds:
                print(p)
        return

    # 4) 交互（默认兜底）
    if args.interactive or (args.text is None and args.input is None and not args.test_pkl):
        print("Interactive mode: 输入中文回车翻译；直接回车退出。")
        while True:
            try:
                zh = input("ZH> ").strip()
            except EOFError:
                break
            if not zh:
                break
            src_ids = encode_zh(zh, zh_word2idx).to(config.device)
            pred = translate_one(
                model, src_ids, en_sp,
                decode=args.decode,
                beam_size=args.beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty
            )
            print("EN> " + pred)


if __name__ == "__main__":
    main()
