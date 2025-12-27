import utils
import config
import logging
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MTDataset
from utils import chinese_tokenizer_load  # Source language is Chinese now
from model import make_model, LabelSmoothing


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Transformer NMT Training with Architecture Ablation')

    # 数据集参数
    parser.add_argument('--data_size', type=str, default=None,
                        help='Dataset size: "10k" or "100k" (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')

    # 架构消融参数
    parser.add_argument('--position_encoding', type=str, default=None,
                        choices=['absolute', 'relative'],
                        help='Position encoding type: absolute or relative (default: from config)')
    parser.add_argument('--norm_type', type=str, default=None,
                        choices=['layernorm', 'rmsnorm'],
                        help='Normalization type: layernorm or rmsnorm (default: from config)')
    parser.add_argument('--max_relative_position', type=int, default=None,
                        help='Maximum relative position for relative position encoding (default: from config)')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=None,
                        help='Model dimension (default: from config)')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Number of attention heads (default: from config)')
    parser.add_argument('--n_layers', type=int, default=None,
                        help='Number of encoder/decoder layers (default: from config)')
    parser.add_argument('--d_ff', type=int, default=None,
                        help='Feed-forward dimension (default: from config)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (default: from config)')

    # 训练参数
    parser.add_argument('--epoch_num', type=int, default=None,
                        help='Number of training epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--use_noamopt', action='store_true', default=None,
                        help='Use Noam optimizer (default: from config)')
    parser.add_argument('--no_noamopt', action='store_true',
                        help='Do not use Noam optimizer')
    parser.add_argument('--use_smoothing', action='store_true', default=None,
                        help='Use label smoothing (default: from config)')
    parser.add_argument('--no_smoothing', action='store_true',
                        help='Do not use label smoothing')

    # 实验标识和输出路径
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name suffix for model/log files (default: architecture config)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Custom path to save model (e.g., ./my_models/my_model.pth)')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Custom path to save training log (e.g., ./my_logs/train.log)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Custom path to save translation output (e.g., ./my_outputs/output.txt)')

    # GPU配置
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU device ID (default: from config)')

    return parser.parse_args()


def update_config_from_args(args):
    """根据命令行参数更新config"""
    # 数据集参数
    if args.data_size is not None:
        config.DATA_SIZE = args.data_size
        # 更新相关路径
        config.data_dir = f'./data_{config.DATA_SIZE}'
        config.train_data_path = f'./data_{config.DATA_SIZE}/json/train.json'
        config.dev_data_path = f'./data_{config.DATA_SIZE}/json/dev.json'
        config.test_data_path = f'./data_{config.DATA_SIZE}/json/test.json'
        config.tokenizer_dir = f'./data_{config.DATA_SIZE}/tokenizer'
        # 更新词汇表大小
        config.src_vocab_size = config.src_vocab_size_10k if config.DATA_SIZE == "10k" else config.src_vocab_size_100k

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # 架构消融参数
    if args.position_encoding is not None:
        config.position_encoding_type = args.position_encoding
    if args.norm_type is not None:
        config.norm_type = args.norm_type
    if args.max_relative_position is not None:
        config.max_relative_position = args.max_relative_position

    # 模型参数
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.dropout is not None:
        config.dropout = args.dropout

    # 训练参数
    if args.epoch_num is not None:
        config.epoch_num = args.epoch_num
    if args.lr is not None:
        config.lr = args.lr
    if args.no_noamopt:
        config.use_noamopt = False
    elif args.use_noamopt is not None:
        config.use_noamopt = args.use_noamopt
    if args.no_smoothing:
        config.use_smoothing = False
    elif args.use_smoothing is not None:
        config.use_smoothing = args.use_smoothing

    # GPU配置
    if args.gpu_id is not None:
        config.gpu_id = args.gpu_id
        if config.gpu_id != '':
            config.device = torch.device(f"cuda:{config.gpu_id}")
        else:
            config.device = torch.device('cpu')

    # 处理输出路径：优先使用用户指定的路径，否则使用exp_name或自动生成
    if args.model_path is not None or args.log_path is not None or args.output_path is not None:
        # 用户指定了自定义路径
        if args.model_path is not None:
            config.model_path = args.model_path
        if args.log_path is not None:
            config.log_path = args.log_path
        if args.output_path is not None:
            config.output_path = args.output_path
    else:
        # 使用exp_name或自动生成路径
        if args.exp_name is not None:
            exp_suffix = args.exp_name
        else:
            # 根据架构配置自动生成名称
            exp_suffix = f"{config.DATA_SIZE}_pos-{config.position_encoding_type}_norm-{config.norm_type}"

        # 创建实验子文件夹
        import os
        exp_dir = f'./experiment/{exp_suffix}'
        os.makedirs(exp_dir, exist_ok=True)

        # 更新模型和输出路径到子文件夹
        config.model_path = f'{exp_dir}/model_{exp_suffix}.pth'
        config.log_path = f'{exp_dir}/train_{exp_suffix}.log'
        config.output_path = f'{exp_dir}/output_{exp_suffix}.txt'


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model, warmup_steps=None):
    """for batch_size 32, ~2565 steps per epoch
    Args:
        model: Transformer model
        warmup_steps: Number of warmup steps (default: from config or 2000)
    """
    if warmup_steps is None:
        warmup_steps = getattr(config, 'warmup_steps', 2000)

    return NoamOpt(model.src_embed[0].d_model, 1, warmup_steps,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                    eps=1e-9))  # ✅ Removed weight_decay (was 0.01, too large!)


def run():
    # 解析命令行参数
    args = parse_args()
    # 根据命令行参数更新配置
    update_config_from_args(args)

    utils.set_logger(config.log_path)

    # 显示当前使用的数据集配置
    logging.info("="*60)
    logging.info(f"Training Configuration")
    logging.info("="*60)
    logging.info(f"Dataset: {config.DATA_SIZE}")
    logging.info(f"Chinese vocab size: {config.src_vocab_size}")
    logging.info(f"English vocab size: {config.tgt_vocab_size}")
    logging.info(f"Data directory: {config.data_dir}")
    logging.info("="*60)
    logging.info(f"Architecture Configuration:")
    logging.info(f"  Position Encoding: {config.position_encoding_type}")
    logging.info(f"  Normalization: {config.norm_type}")
    if config.position_encoding_type == 'relative':
        logging.info(f"  Max Relative Position: {config.max_relative_position}")
    logging.info(f"  Model dimension: {config.d_model}")
    logging.info(f"  Attention heads: {config.n_heads}")
    logging.info(f"  Layers: {config.n_layers}")
    logging.info(f"  Feed-forward dim: {config.d_ff}")
    logging.info(f"  Dropout: {config.dropout}")
    logging.info("="*60)
    logging.info(f"Training Configuration:")
    logging.info(f"  Batch size: {config.batch_size}")
    logging.info(f"  Epochs: {config.epoch_num}")
    logging.info(f"  Learning rate: {config.lr}")
    logging.info(f"  Use NoamOpt: {config.use_noamopt}")
    logging.info(f"  Use Label Smoothing: {config.use_smoothing}")
    logging.info("="*60)
    logging.info(f"Model will be saved to: {config.model_path}")
    logging.info("="*60)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型 - 传递架构消融参数
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
        max_relative_position=config.max_relative_position
    )
    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataloader, dev_dataloader, test_dataloader, model, model_par, criterion, optimizer)
    # Note: test() function is now called at the end of train() for final evaluation


def check_opt():
    """check learning rate changes"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)
    # Three settings of the lrate hyperparameters.
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sent, beam_search=True):
    """单句翻译：中文 → 英文"""
    # 初始化模型 - 传递架构消融参数
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
        max_relative_position=config.max_relative_position
    )
    sp_chn = chinese_tokenizer_load()  # Chinese tokenizer for source
    BOS = sp_chn.bos_id()  # 2
    EOS = sp_chn.eos_id()  # 3
    src_tokens = [[BOS] + sp_chn.EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """单句翻译示例：中文 → 英文"""
    sent = "随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。"
    # Expected English: "As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening."
    one_sentence_translate(sent, beam_search=True)


if __name__ == "__main__":
    import os
    # Single GPU configuration - set to your available GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    import warnings
    warnings.filterwarnings('ignore')
    run()  # Start training
    # translate_example()  # Uncomment for single sentence translation
