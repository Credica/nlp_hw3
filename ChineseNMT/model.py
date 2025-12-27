import config
from data_loader import subsequent_mask

import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative Positional Encoding

    Instead of adding absolute position embeddings to the input,
    this adds relative position information to the attention scores.
    Based on "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    and T5's simplified relative position bias.
    """
    def __init__(self, d_model, dropout, max_len=5000, num_heads=8):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len

        # 相对位置编码不需要在embedding层添加位置信息
        # 位置信息将在attention层中通过bias的形式添加

    def forward(self, x):
        # 对于相对位置编码，我们只需要对输入进行dropout
        # 不添加任何位置信息（位置信息在attention中处理）
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None, position_bias=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果使用相对位置编码，添加position bias
    if position_bias is not None:
        scores = scores + position_bias

    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, use_relative_position=False, max_relative_position=32):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # 相对位置编码相关
        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.max_relative_position = max_relative_position
            # 为每个head创建相对位置的embedding
            # 位置范围: [-max_relative_position, max_relative_position]
            vocab_size = max_relative_position * 2 + 1
            self.relative_position_embeddings = nn.Embedding(vocab_size, self.d_k)

    def _get_relative_position_bias(self, query_len, key_len):
        """计算相对位置bias

        Args:
            query_len: Length of query sequence
            key_len: Length of key sequence
        """
        # 创建query和key的位置向量
        query_range = torch.arange(query_len, device=DEVICE)
        key_range = torch.arange(key_len, device=DEVICE)
        # 创建相对位置矩阵 [query_len, key_len]
        # 对每个query position i和key position j，计算相对位置 (i - j)
        relative_position_matrix = query_range.unsqueeze(1) - key_range.unsqueeze(0)
        # 裁剪到 [-max_relative_position, max_relative_position]
        relative_position_matrix = torch.clamp(
            relative_position_matrix,
            -self.max_relative_position,
            self.max_relative_position
        )
        # 转换为正数索引 [0, 2*max_relative_position]
        relative_position_matrix = relative_position_matrix + self.max_relative_position
        # 获取相对位置embeddings并调整形状
        # [query_len, key_len, d_k]
        relative_embeddings = self.relative_position_embeddings(relative_position_matrix)
        # [query_len, key_len, d_k] -> [d_k, query_len, key_len]
        relative_embeddings = relative_embeddings.permute(2, 0, 1)
        # 使用sum将d_k维度合并，得到[query_len, key_len]
        relative_bias = relative_embeddings.sum(dim=0)
        relative_bias = relative_bias.unsqueeze(0)  # [1, query_len, key_len]
        relative_bias = relative_bias.expand(self.h, -1, -1)  # [h, query_len, key_len]
        return relative_bias.unsqueeze(0)  # [1, h, query_len, key_len]

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        # 计算相对位置bias（如果使用）
        position_bias = None
        if self.use_relative_position:
            # [1, h, query_len, key_len]
            rel_bias = self._get_relative_position_bias(query_len, key_len)
            # 扩展到batch维度: [nbatches, h, query_len, key_len]
            position_bias = rel_bias.expand(nbatches, -1, -1, -1)

        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, position_bias=position_bias)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 返回Layer Norm的结果
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    RMSNorm simplifies LayerNorm by removing the mean centering operation
    and only normalizing by the root mean square.
    Paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, features, eps=1e-6):
        super(RMSNorm, self).__init__()
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 计算RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return self.scale * x / rms


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    def __init__(self, size, dropout, norm_type='layernorm'):
        super(SublayerConnection, self).__init__()
        # 根据norm_type选择归一化方法
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(size)
        else:  # 默认使用layernorm
            self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N, norm_type='layernorm'):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm - 根据norm_type选择
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(layer.size)
        else:
            self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, norm_type='layernorm'):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(size, dropout, norm_type), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N, norm_type='layernorm'):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm - 根据norm_type选择
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(layer.size)
        else:
            self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, norm_type='layernorm'):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm_type), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1,
               position_encoding_type='absolute', norm_type='layernorm',
               use_relative_position=False, max_relative_position=32):
    """
    构建Transformer模型

    Args:
        src_vocab: 源语言词汇表大小
        tgt_vocab: 目标语言词汇表大小
        N: encoder和decoder的层数
        d_model: 模型维度
        d_ff: feed-forward层的维度
        h: 多头注意力的头数
        dropout: dropout比率
        position_encoding_type: 位置编码类型，'absolute' 或 'relative'
        norm_type: 归一化类型，'layernorm' 或 'rmsnorm'
        use_relative_position: 是否在attention中使用相对位置bias（仅当position_encoding_type='relative'时有效）
        max_relative_position: 相对位置的最大距离
    """
    c = copy.deepcopy

    # 根据position_encoding_type决定是否在attention中使用相对位置
    if position_encoding_type == 'relative':
        use_relative_position = True

    # 实例化Attention对象（支持相对位置编码）
    attn = MultiHeadedAttention(h, d_model, dropout, use_relative_position, max_relative_position).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)

    # 根据position_encoding_type选择位置编码方式
    if position_encoding_type == 'relative':
        position = RelativePositionalEncoding(d_model, dropout, max_len=5000, num_heads=h).to(DEVICE)
    else:  # absolute
        position = PositionalEncoding(d_model, dropout).to(DEVICE)

    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout, norm_type).to(DEVICE),
            N,
            norm_type
        ).to(DEVICE),
        Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, norm_type).to(DEVICE),
            N,
            norm_type
        ).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)
    ).to(DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    memory = model.encode(src, src_mask)
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    for s in range(max_len):
        tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src.data)
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)

        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            # print(stop_flag[i])
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results


def greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """传入一个训练好的模型，对指定数据进行预测"""
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        if next_word == end_symbol:
            break
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
