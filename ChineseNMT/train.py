import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import english_tokenizer_load  # Changed to English for target language


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, test_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer, model=model))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None, model=model))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            early_stop = config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break

    # 训练结束后，在测试集上进行最终评估
    logging.info("\n" + "="*60)
    logging.info("Training completed! Evaluating on test set...")
    logging.info("="*60)

    # 加载最佳模型
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    # 在测试集上计算BLEU分数
    test_bleu_score = evaluate(test_data, model, mode='test', use_beam=True)
    logging.info("\nFinal Test Set BLEU Score: {:.2f}".format(test_bleu_score))

    # 同时也在测试集上测试greedy解码的性能
    test_bleu_greedy = evaluate(test_data, model, mode='test', use_beam=False)
    logging.info("Test Set BLEU Score (Greedy): {:.2f}".format(test_bleu_greedy))
    logging.info("Improvement with Beam Search: {:.2f} BLEU points".format(test_bleu_score - test_bleu_greedy))
    logging.info("="*60)

    return test_bleu_score


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            # Gradient clipping REMOVED for small dataset (10k)
            # Small datasets need more flexible training
            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
            self.opt.step()
            # Use appropriate zero_grad based on optimizer type
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5, model=None):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        self.model = model  # Store model reference for gradient clipping

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            # Gradient clipping REMOVED for small dataset (10k)
            # Small datasets need more flexible training
            # if self.model is not None:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            # Use appropriate zero_grad based on optimizer type
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize


def evaluate(data, model, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_eng = english_tokenizer_load()  # English tokenizer for target language
    trg = []
    res = []
    with torch.no_grad():
        # 在data的中文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的英文句子（目标语言）
            en_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
                # beam search 返回 [[候选1, 候选2, ...], ...]，取每个样本的第一个候选
                decode_result = [h[0] for h in decode_result]
            else:
                # greedy decode 直接返回 [[token_ids], ...]，不需要额外处理
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            translation = [sp_eng.decode_ids(_s) for _s in decode_result]
            trg.extend(en_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w", encoding="utf-8") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    # Using BLEU-4 as required by assignment (sacrebleu default is BLEU-4)
    # '13a' tokenizer is appropriate for English (target language)
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='13a')
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_eng = english_tokenizer_load()  # English tokenizer for target language
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            # beam search 返回 [[候选1, 候选2, ...], ...]，取每个样本的第一个候选
            decode_result = [h[0] for h in decode_result]
        else:
            # greedy decode 直接返回 [[token_ids], ...]，不需要额外处理
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_eng.decode_ids(_s) for _s in decode_result]
        print(translation[0])
