#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer.py - 中文 -> 英文 推理脚本（支持 beam / greedy，支持单句、文件、交互模式）

用法示例：
1) 单句：
   python infer.py --text "随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件。"

2) 文件（每行一句中文）：
   python infer.py --input zh.txt --output en.txt

3) 交互：
   python infer.py --interactive

4) 使用特定架构的模型（relative position encoding + RMSNorm）：
   python infer.py --interactive --data_size 100k --ckpt path/to/model.pth \
                   --position_encoding relative --norm_type rmsnorm

常用参数：
- --ckpt: 模型权重路径（默认用 config.model_path）
- --data_size: 10k 或 100k（会同步更新 config 的路径/词表大小）
- --gpu_id: 指定 GPU（例如 "0"），或 --cpu 强制 CPU
- --decode: beam 或 greedy
- --position_encoding: absolute 或 relative（必须与训练时一致）
- --norm_type: layernorm 或 rmsnorm（必须与训练时一致）
"""

import os
import sys
import argparse
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

import config
from model import make_model, batch_greedy_decode
from beam_decoder import beam_search
from utils import chinese_tokenizer_load, english_tokenizer_load


def update_config_for_infer(args: argparse.Namespace) -> None:
    """根据命令行参数更新 config（只更新推理需要的部分）"""
    # 数据集大小影响：data_dir / tokenizer_dir / src_vocab_size / 默认 ckpt 路径
    if args.data_size is not None:
        config.DATA_SIZE = args.data_size
        config.data_dir = f'./data_{config.DATA_SIZE}'
        config.train_data_path = f'./data_{config.DATA_SIZE}/json/train.json'
        config.dev_data_path = f'./data_{config.DATA_SIZE}/json/dev.json'
        config.test_data_path = f'./data_{config.DATA_SIZE}/json/test.json'
        config.tokenizer_dir = f'./data_{config.DATA_SIZE}/tokenizer'
        config.src_vocab_size = config.src_vocab_size_10k if config.DATA_SIZE == "10k" else config.src_vocab_size_100k

        # 也同步默认输出/模型路径（和你的 config 风格一致）
        config.model_path = f'./experiment/model_{config.DATA_SIZE}.pth'
        config.output_path = f'./experiment/output_{config.DATA_SIZE}.txt'

    # 覆盖模型架构参数
    if args.position_encoding is not None:
        config.position_encoding_type = args.position_encoding
    if args.norm_type is not None:
        config.norm_type = args.norm_type
    if args.max_relative_position is not None:
        config.max_relative_position = args.max_relative_position

    # 覆盖解码参数
    if args.max_len is not None:
        config.max_len = args.max_len
    if args.beam_size is not None:
        config.beam_size = args.beam_size

    # 设备选择
    if args.cpu:
        config.gpu_id = ''
        config.device = torch.device('cpu')
    elif args.gpu_id is not None:
        config.gpu_id = args.gpu_id
        config.device = torch.device(f"cuda:{config.gpu_id}") if config.gpu_id != '' else torch.device('cpu')


def build_model() -> torch.nn.Module:
    """按 config 构建模型结构（必须与训练时一致）"""
    model = make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_heads,
        config.dropout,
        position_encoding_type=config.position_encoding_type,
        norm_type=config.norm_type,
        max_relative_position=getattr(config, "max_relative_position", 32),
    )
    return model


def _strip_module_prefix(state_dict: dict) -> dict:
    """兼容 checkpoint 里 key 带 'module.' 的情况"""
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = _strip_module_prefix(ckpt)
    model.load_state_dict(ckpt, strict=True)


def encode_batch_texts(
    texts: List[str],
    sp_chn,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把一批中文句子 -> padding 后的 src tensor + src_mask"""
    BOS = sp_chn.bos_id()
    EOS = sp_chn.eos_id()
    PAD = sp_chn.pad_id()

    token_lists = []
    for t in texts:
        ids = [BOS] + sp_chn.EncodeAsIds(t) + [EOS]
        token_lists.append(torch.LongTensor(ids))

    src = pad_sequence(token_lists, batch_first=True, padding_value=PAD).to(device)
    src_mask = (src != PAD).unsqueeze(-2)  # [B, 1, S]
    return src, src_mask


def decode_batch(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    sp_eng,
    decode_mode: str = "beam",
) -> List[str]:
    """对一批 src 做翻译，返回英文字符串列表"""
    model.eval()
    with torch.no_grad():
        if decode_mode == "beam":
            hyps, _ = beam_search(
                model,
                src,
                src_mask,
                config.max_len,
                config.padding_idx,
                config.bos_idx,
                config.eos_idx,
                config.beam_size,
                config.device,
            )
            # 取每个样本的 top-1
            out_ids_list = [h[0] for h in hyps]
        else:
            out_ids_list = batch_greedy_decode(
                model, src, src_mask, max_len=config.max_len
            )

    # SentencePiece decode
    translations = [sp_eng.decode_ids(ids) for ids in out_ids_list]
    return translations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Chinese->English inference")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: config.model_path)")
    p.add_argument("--data_size", type=str, default=None, choices=["10k", "100k"], help="10k or 100k")
    p.add_argument("--gpu_id", type=str, default=None, help='GPU id, e.g. "0"')
    p.add_argument("--cpu", action="store_true", help="force CPU")

    # Model architecture parameters
    p.add_argument("--position_encoding", type=str, default=None, choices=["absolute", "relative"],
                   help="position encoding type (default: config.position_encoding_type)")
    p.add_argument("--norm_type", type=str, default=None, choices=["layernorm", "rmsnorm"],
                   help="normalization type (default: config.norm_type)")
    p.add_argument("--max_relative_position", type=int, default=None,
                   help="max relative position (only for relative position encoding)")

    p.add_argument("--decode", type=str, default="beam", choices=["beam", "greedy"], help="decode mode")
    p.add_argument("--beam_size", type=int, default=None, help="beam size (only for beam)")
    p.add_argument("--max_len", type=int, default=None, help="max decoding length")

    p.add_argument("--text", type=str, default=None, help="single sentence to translate")
    p.add_argument("--input", type=str, default=None, help="input file, one Chinese sentence per line")
    p.add_argument("--output", type=str, default=None, help="output file (optional)")
    p.add_argument("--batch", type=int, default=32, help="batch size for file translation")
    p.add_argument("--interactive", action="store_true", help="interactive mode")
    return p.parse_args()


def main():
    args = parse_args()
    update_config_for_infer(args)

    ckpt_path = args.ckpt or config.model_path

    # 加载分词器（会用到 config.tokenizer_dir）
    sp_chn = chinese_tokenizer_load()
    sp_eng = english_tokenizer_load()

    # 构建并加载模型
    model = build_model().to(config.device)
    load_checkpoint(model, ckpt_path, config.device)
    model.eval()

    # 1) 单句
    if args.text is not None:
        src, src_mask = encode_batch_texts([args.text], sp_chn, config.device)
        out = decode_batch(model, src, src_mask, sp_eng, decode_mode=args.decode)
        print(out[0])
        return

    # 2) 文件翻译
    if args.input is not None:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        with open(args.input, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        lines = [ln for ln in lines if ln]  # 去空行

        outputs: List[str] = []
        bs = max(1, args.batch)
        for i in range(0, len(lines), bs):
            chunk = lines[i:i + bs]
            src, src_mask = encode_batch_texts(chunk, sp_chn, config.device)
            outs = decode_batch(model, src, src_mask, sp_eng, decode_mode=args.decode)
            outputs.extend(outs)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for zh, en in zip(lines, outputs):
                    f.write(en + "\n")
            print(f"[OK] Saved to: {args.output}")
        else:
            for en in outputs:
                print(en)
        return

    # 3) 交互模式
    if args.interactive or (args.text is None and args.input is None):
        print("Interactive mode. 输入中文句子回车翻译；输入空行退出。")
        while True:
            try:
                sent = input("ZH> ").strip()
            except EOFError:
                break
            if not sent:
                break
            src, src_mask = encode_batch_texts([sent], sp_chn, config.device)
            out = decode_batch(model, src, src_mask, sp_eng, decode_mode=args.decode)
            print("EN> " + out[0])
        return


if __name__ == "__main__":
    main()
