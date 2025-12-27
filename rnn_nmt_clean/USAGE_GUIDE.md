# RNN NMT - 使用指南

## 修改总结

✅ **已完成的修改**：

### 1. 数据集选择功能
- 支持通过 `--dataset` 参数选择 10k 或 100k 数据集
- 每个数据集有独立的目录：
  - `tokenizer_10k/` 和 `checkpoints_10k/` 用于 10k 数据集
  - `tokenizer_100k/` 和 `checkpoints_100k/` 用于 100k 数据集

### 2. 运行方式优化
- 移除了交互式选择菜单
- 直接运行 `python run_experiments.py` 进行快速对比
- 支持命令行参数 `--dataset 10k` 或 `--dataset 100k`

### 3. Beam Search 测试
- 现在测试 beam sizes: [1, 3, 5, 10]
- 1 = greedy decoding (贪婪解码)
- 3, 5, 10 = 不同大小的 beam search

### 4. 修复的问题
- 修复了 `visualize_results.py` 中的键名不匹配问题
- 修复了 `preprocess.py` 的导入问题
- 修复了路径生成问题

---

## 使用流程

### 步骤 1: 预处理数据

**预处理 10k 数据集**：
```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate nlp
python preprocess.py --dataset 10k
```

**预处理 100k 数据集**：
```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate nlp
python preprocess.py --dataset 100k
```

**注意**：100k 数据集预处理时间更长，需要更多内存。

### 步骤 2: 运行实验

**使用 10k 数据集进行完整实验**：
```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate nlp
python run_experiments.py --dataset 10k
```

**使用 100k 数据集进行完整实验**：
```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate nlp
python run_experiments.py --dataset 100k
```

**运行快速对比实验**：
```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate nlp
python -c "from run_experiments import quick_comparison; quick_comparison()"
```

### 步骤 3: 查看结果
- 实验结果保存在 `checkpoints_10k/` 或 `checkpoints_100k/`
- 可视化图表保存在 `checkpoints_10k/plots/` 或 `checkpoints_100k/plots/`
- 文本结果保存在 `checkpoints_10k/experiment_results.txt` 或 `checkpoints_100k/experiment_results.txt`

---

## 快速对比 vs 完整实验

### 快速对比 (Quick Comparison)
- **配置数量**: 5 个
- **训练时长**: ~15-30 分钟 (10k 数据集)
- **配置**:
  1. dot attention + 50% TF
  2. multiplicative attention + 50% TF
  3. additive attention + 50% TF
  4. additive attention + 100% TF
  5. additive attention + 0% TF
- **测试**: 每种配置只测试 beam-3
- **运行方式**: `python -c "from run_experiments import quick_comparison; quick_comparison()"`

### 完整实验 (Comprehensive Experiments) - **默认模式**
- **配置数量**: 9 个或更多
- **训练时长**: ~1-2 小时 (10k 数据集)
- **Phase 1**: 3 种 attention × 50% TF (3 个模型)
- **Phase 2**: additive attention × 3 种 TF 策略 (3 个模型)
- **Phase 3**: 测试所有 beam sizes [1, 3, 5, 10]
- **测试**: 每种配置测试所有 beam sizes
- **运行方式**: `python run_experiments.py --dataset 10k`

---

## 文件结构

```
rnn_nmt_clean/
├── tokenizer_10k/              # 10k 数据集的 tokenizer
│   ├── zh_vocab.pkl
│   ├── eng.model
│   ├── train_data.pkl
│   ├── valid_data.pkl
│   ├── test_data.pkl
│   └── config_info.pkl
├── tokenizer_100k/             # 100k 数据集的 tokenizer
│   ├── zh_vocab.pkl
│   ├── eng.model
│   ├── train_data.pkl
│   ├── valid_data.pkl
│   ├── test_data.pkl
│   └── config_info.pkl
├── checkpoints_10k/            # 10k 数据集的模型和结果
│   ├── model_*.pt
│   ├── experiment_results.txt
│   └── plots/
├── checkpoints_100k/           # 100k 数据集的模型和结果
│   ├── model_*.pt
│   ├── experiment_results.txt
│   └── plots/
├── run_experiments.py          # 快速对比实验脚本
├── experiments.py              # 完整实验脚本
├── preprocess.py               # 数据预处理脚本
├── config_v2.py                # 配置（支持数据集选择）
├── beam_search.py              # Beam search 实现（有问题）
└── visualize_results.py        # 结果可视化脚本
```

---

## 已知问题

### 1. Beam Search 实现问题
在 `beam_search.py` 中存在一个**严重缺陷**：

**问题位置**: 第 128-131 行
```python
log_probs_expanded = log_probs.expand(beam.size, -1)
```

**问题描述**: 所有 beam 在每一步都使用完全相同的概率分布，每个 beam 应该：
- 有自己的 hidden state
- 基于自己的历史计算下一步
- 有不同的概率分布

**当前行为**: 所有 beam 行为相同，导致 beam-3 和 beam-5 结果相同

**建议修复**: 为每个 beam 维护独立的 hidden state，并在每一步为每个 beam 分别计算输出

### 2. NumPy 版本兼容性
可能遇到 NumPy 版本警告，但不影响功能：
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**解决方案**: 升级相关包或降级 NumPy：
```bash
pip install numpy<2  # 降级到 NumPy 1.x
```

---

## 实验结果示例

根据 `checkpoints/experiment_results.txt` 的内容：

### 注意力机制对比 (Teacher Forcing = 50%)
- **DOT Attention**: BLEU = 6.58
- **MULTIPLICATIVE Attention**: BLEU = 7.78
- **ADDITIVE Attention**: BLEU = 8.16

### 训练策略对比 (Additive Attention)
- **50% (Mixed)**: BLEU = 8.16
- **100% (Teacher Forcing)**: BLEU = 8.23
- **0% (Free Running)**: BLEU = 6.59

### 解码策略对比
- **Greedy (Beam-1)**: BLEU = 8.16
- **Beam-3**: BLEU = 3.99 (存在问题)
- **Beam-5**: BLEU = 3.99 (存在问题)
- **Beam-10**: BLEU = 3.99 (存在问题)

**注意**: 由于 beam search 实现问题，beam-3、beam-5 和 beam-10 的结果都异常低。

---

## 性能优化建议

1. **使用 GPU**: 确保 CUDA 可用
2. **调整 batch size**: 在 `config_v2.py` 中修改 `BATCH_SIZE`
3. **减少 epoch 数**: 在 `run_experiments.py` 中将 `epochs=10` 改为更小的值
4. **使用 10k 数据集**: 用于快速测试，100k 用于最终实验

---

## 常用命令

```bash
# 激活环境
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate nlp

# 预处理 10k 数据
python preprocess.py --dataset 10k

# 运行完整实验（默认模式）
python run_experiments.py --dataset 10k

# 运行快速对比实验
python -c "from run_experiments import quick_comparison; quick_comparison()"

# 检查结果
cat checkpoints_10k/experiment_results.txt

# 查看训练曲线
ls checkpoints_10k/plots/
```
