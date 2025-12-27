# Architecture Ablation Experiments Guide

本指南说明如何运行架构消融实验（Position Encoding 和 Normalization）。

## 🎯 实验目标
根据作业要求，我们需要完成以下架构消融实验：

1. **位置编码对比**: Absolute Position Encoding vs Relative Position Encoding
2. **归一化方法对比**: LayerNorm vs RMSNorm

## 📋 已实现的功能

### 1. 位置编码变体
- **Absolute Position Encoding** (原始实现)
  - 使用正弦/余弦函数生成固定的位置编码
  - 在embedding层添加位置信息

- **Relative Position Encoding** (新增)
  - 在attention层中添加相对位置bias
  - 基于T5和Transformer-XL的简化实现
  - 可配置最大相对位置距离

### 2. 归一化方法
- **LayerNorm** (原始实现)
  - 标准的Layer Normalization
  - 包含均值和方差归一化

- **RMSNorm** (新增)
  - Root Mean Square Normalization
  - 只进行RMS归一化，去除了均值中心化
  - 参数更少，计算更快

### 3. 命令行参数支持
所有配置参数现在都可以通过命令行传入，无需修改配置文件。

## 🚀 快速开始

### 方式一：运行完整的消融实验

运行所有4个架构组合的实验：

```bash
cd /mnt/c/Users/sysu/Desktop/nlp_hw3/ChineseNMT
bash run_ablation_experiments.sh
```

这将依次运行：
1. Baseline (Absolute + LayerNorm)
2. Relative Position + LayerNorm
3. Absolute Position + RMSNorm
4. Relative Position + RMSNorm

### 方式二：运行单个实验

使用 `run_single_experiment.sh` 或直接调用 `main.py`：

#### 示例1: 测试相对位置编码
```bash
python main.py --position_encoding relative --data_size 10k --epoch_num 40
```

#### 示例2: 测试RMSNorm
```bash
python main.py --norm_type rmsnorm --data_size 10k --epoch_num 40
```

#### 示例3: 组合测试（相对位置 + RMSNorm）
```bash
python main.py \
    --position_encoding relative \
    --norm_type rmsnorm \
    --data_size 10k \
    --epoch_num 40 \
    --exp_name "test_relative_rms"
```

#### 示例4: 自定义模型大小
```bash
python main.py \
    --d_model 256 \
    --n_heads 4 \
    --n_layers 4 \
    --d_ff 1024 \
    --data_size 10k \
    --exp_name "small_model"
```

## 📊 命令行参数详解

### 架构消融参数
```bash
--position_encoding TYPE    # 位置编码类型: absolute 或 relative
--norm_type TYPE           # 归一化类型: layernorm 或 rmsnorm
--max_relative_position N  # 相对位置编码的最大距离 (默认: 32)
```

### 模型架构参数
```bash
--d_model N       # 模型维度 (默认: 512)
--n_heads N       # 注意力头数 (默认: 8)
--n_layers N      # 编码器/解码器层数 (默认: 6)
--d_ff N          # Feed-forward维度 (默认: 2048)
--dropout RATE    # Dropout比率 (默认: 0.1)
```

### 训练参数
```bash
--data_size SIZE     # 数据集大小: 10k 或 100k
--batch_size N       # Batch大小 (默认: 32)
--epoch_num N        # 训练轮数 (默认: 40)
--lr RATE           # 学习率 (默认: 1e-4)
--use_noamopt       # 使用Noam优化器
--no_noamopt        # 不使用Noam优化器
--use_smoothing     # 使用Label Smoothing
--no_smoothing      # 不使用Label Smoothing
```

### 其他参数
```bash
--exp_name NAME       # 实验名称（自动生成路径）
--model_path PATH     # 自定义模型保存路径
--log_path PATH       # 自定义日志保存路径
--output_path PATH    # 自定义翻译输出路径
--gpu_id ID          # GPU设备ID (默认: 0)
```

## 📁 路径和文件命名

### 方式1: 使用 `--exp_name`（推荐）

自动在 `experiment/` 目录下生成文件：

```bash
python main.py --exp_name "my_experiment"
```

生成文件：
- `./experiment/model_my_experiment.pth`
- `./experiment/train_my_experiment.log`
- `./experiment/output_my_experiment.txt`

### 方式2: 完全自定义路径

```bash
python main.py \
    --model_path "./my_models/transformer_v1.pth" \
    --log_path "./my_logs/training.log" \
    --output_path "./my_results/output.txt"
```

### 方式3: 部分自定义

```bash
# 只自定义模型路径，其他使用exp_name
python main.py \
    --exp_name "test" \
    --model_path "./saved_models/best_model.pth"
```

### 方式4: 自动生成（推荐用于对比实验）

不指定任何路径参数，根据架构配置自动命名：

```bash
python main.py --position_encoding relative --norm_type rmsnorm
```

生成: `model_10k_pos-relative_norm-rmsnorm.pth`

**更多路径使用示例，请查看**: `CUSTOM_PATH_EXAMPLES.md`

## 📁 实验结果文件

运行实验后，结果将保存在 `./experiment/` 目录：

```
experiment/
├── model_10k_baseline_abs_ln.pth       # 模型权重
├── train_10k_baseline_abs_ln.log       # 训练日志
├── output_10k_baseline_abs_ln.txt      # 翻译结果
├── model_10k_relative_ln.pth
├── train_10k_relative_ln.log
├── ...
```

## 🔍 查看实验结果

### 1. 查看训练日志
```bash
tail -f experiment/train_10k_baseline_abs_ln.log
```

### 2. 对比不同实验的BLEU分数
```bash
grep "Bleu Score" experiment/train_*.log
```

### 3. 查看最终测试结果
```bash
grep "Test" experiment/train_*.log
```

## 📝 实验建议

### 1. 快速验证（使用10k数据集）
适合快速测试架构变体，验证实现正确性：
```bash
python main.py \
    --data_size 10k \
    --epoch_num 20 \
    --position_encoding relative
```

### 2. 完整实验（使用100k数据集）
用于最终的性能对比和论文撰写：
```bash
python main.py \
    --data_size 100k \
    --epoch_num 40 \
    --position_encoding relative \
    --exp_name "final_relative_100k"
```

### 3. 超参数敏感性实验

#### Batch Size实验
```bash
for bs in 8 16 32 64; do
    python main.py --batch_size $bs --data_size 10k --exp_name "bs_${bs}"
done
```

#### Learning Rate实验
```bash
for lr in 1e-5 5e-5 1e-4 5e-4; do
    python main.py --lr $lr --no_noamopt --data_size 10k --exp_name "lr_${lr}"
done
```

#### Model Scale实验
```bash
# Small model
python main.py --d_model 256 --n_heads 4 --n_layers 4 --d_ff 1024 \
    --data_size 10k --exp_name "small"

# Base model (default)
python main.py --data_size 10k --exp_name "base"

# Large model
python main.py --d_model 768 --n_heads 12 --n_layers 8 --d_ff 3072 \
    --data_size 10k --exp_name "large"
```

## 🐛 故障排除

### 问题1: CUDA out of memory
**解决方案**: 减小batch size或模型维度
```bash
python main.py --batch_size 16 --d_model 256
```

### 问题2: 词汇表大小不匹配
**解决方案**: 确保数据预处理已完成
```bash
python preprocess_data.py
```

### 问题3: 找不到数据文件
**解决方案**: 检查数据路径和DATA_SIZE设置
```bash
ls -la data_10k/json/
```

## 📊 实验对比分析

完成所有实验后，建议从以下角度进行对比分析：

1. **定量指标**
   - BLEU分数
   - 训练时间（每个epoch）
   - 收敛速度（达到目标BLEU的epoch数）
   - 模型参数量

2. **定性分析**
   - 长句子翻译质量
   - 训练稳定性
   - 过拟合情况

3. **可视化**
   - 训练曲线对比
   - Loss变化趋势
   - BLEU分数变化

## 🎓 代码结构

```
ChineseNMT/
├── model.py                           # 模型定义（已修改）
│   ├── PositionalEncoding             # 绝对位置编码
│   ├── RelativePositionalEncoding     # 相对位置编码（新增）
│   ├── LayerNorm                      # LayerNorm
│   ├── RMSNorm                        # RMSNorm（新增）
│   └── make_model()                   # 支持架构变体（已修改）
├── config.py                          # 配置文件（已修改）
│   ├── position_encoding_type         # 新增
│   ├── norm_type                      # 新增
│   └── max_relative_position          # 新增
├── main.py                            # 主程序（已修改）
│   ├── parse_args()                   # 命令行参数解析（新增）
│   ├── update_config_from_args()      # 配置更新（新增）
│   └── run()                          # 训练主函数（已修改）
├── run_ablation_experiments.sh        # 完整消融实验脚本（新增）
└── run_single_experiment.sh           # 单实验运行脚本（新增）
```

## ✅ 下一步

1. **运行实验**: 使用 `run_ablation_experiments.sh` 运行所有实验
2. **分析结果**: 对比不同架构变体的性能
3. **撰写报告**: 根据实验结果完成报告的架构消融部分
4. **可视化**: 生成训练曲线和性能对比图表


