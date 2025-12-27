# 自定义路径和命名示例

本文档展示如何使用命令行参数自定义模型、日志和输出文件的路径和名称。

---

## 📁 新增的命令行参数

```bash
--exp_name NAME       # 实验名称（会自动在experiment/目录下创建文件）
--model_path PATH     # 完全自定义模型保存路径
--log_path PATH       # 完全自定义日志保存路径
--output_path PATH    # 完全自定义翻译输出路径
```

---

## 🎯 使用场景和示例

### 场景1: 使用实验名称（推荐）

**适用于**: 标准实验，想要统一管理在 experiment/ 目录下

```bash
python main.py \
    --position_encoding relative \
    --data_size 10k \
    --epoch_num 40 \
    --exp_name "relative_position_test_v1"
```

**生成的文件**:
```
experiment/
├── model_relative_position_test_v1.pth
├── train_relative_position_test_v1.log
└── output_relative_position_test_v1.txt
```

---

### 场景2: 完全自定义路径

**适用于**: 需要将文件保存到特定位置，或使用特殊的命名规则

```bash
python main.py \
    --position_encoding relative \
    --data_size 10k \
    --epoch_num 40 \
    --model_path "./saved_models/2025-12-18_transformer_relative.pth" \
    --log_path "./logs/experiment_001.log" \
    --output_path "./results/translations_v1.txt"
```

**生成的文件**:
```
saved_models/
└── 2025-12-18_transformer_relative.pth

logs/
└── experiment_001.log

results/
└── translations_v1.txt
```

**注意**: 需要先创建目录
```bash
mkdir -p saved_models logs results
```

---

### 场景3: 部分自定义

**适用于**: 只想自定义某些文件的路径，其他使用默认

#### 示例3.1: 只自定义模型路径

```bash
python main.py \
    --position_encoding relative \
    --exp_name "test_v1" \
    --model_path "./my_best_models/transformer_checkpoint.pth"
```

**生成的文件**:
```
my_best_models/
└── transformer_checkpoint.pth         # 自定义路径

experiment/
├── train_test_v1.log                 # 使用exp_name
└── output_test_v1.txt                # 使用exp_name
```

#### 示例3.2: 自定义模型和日志路径

```bash
python main.py \
    --position_encoding relative \
    --model_path "./models/model_$(date +%Y%m%d_%H%M%S).pth" \
    --log_path "./logs/train_$(date +%Y%m%d_%H%M%S).log"
```

**生成的文件**:
```
models/
└── model_20251218_143022.pth

logs/
└── train_20251218_143022.log

experiment/
└── output_10k_pos-relative_norm-layernorm.txt  # 自动生成
```

---

### 场景4: 按日期组织实验

**适用于**: 长期项目，需要按时间归档实验

```bash
DATE=$(date +%Y%m%d)
EXP_DIR="./experiments/${DATE}"
mkdir -p ${EXP_DIR}

python main.py \
    --position_encoding relative \
    --norm_type rmsnorm \
    --data_size 10k \
    --model_path "${EXP_DIR}/model.pth" \
    --log_path "${EXP_DIR}/train.log" \
    --output_path "${EXP_DIR}/output.txt"
```

**生成的文件**:
```
experiments/
└── 20251218/
    ├── model.pth
    ├── train.log
    └── output.txt
```

---

### 场景5: 按架构变体组织

**适用于**: 架构消融实验，便于对比

```bash
# 为每个架构变体创建目录
for pos in absolute relative; do
    for norm in layernorm rmsnorm; do
        VARIANT_DIR="./experiments/variants/${pos}_${norm}"
        mkdir -p ${VARIANT_DIR}

        python main.py \
            --position_encoding ${pos} \
            --norm_type ${norm} \
            --data_size 10k \
            --epoch_num 40 \
            --model_path "${VARIANT_DIR}/model.pth" \
            --log_path "${VARIANT_DIR}/train.log" \
            --output_path "${VARIANT_DIR}/output.txt"
    done
done
```

**生成的文件结构**:
```
experiments/variants/
├── absolute_layernorm/
│   ├── model.pth
│   ├── train.log
│   └── output.txt
├── absolute_rmsnorm/
│   ├── model.pth
│   ├── train.log
│   └── output.txt
├── relative_layernorm/
│   ├── model.pth
│   ├── train.log
│   └── output.txt
└── relative_rmsnorm/
    ├── model.pth
    ├── train.log
    └── output.txt
```

---

### 场景6: 使用版本号管理

**适用于**: 迭代优化，需要保留历史版本

```bash
VERSION="v3.2"
python main.py \
    --position_encoding relative \
    --data_size 100k \
    --epoch_num 40 \
    --model_path "./models/transformer_${VERSION}.pth" \
    --log_path "./logs/train_${VERSION}.log" \
    --output_path "./results/output_${VERSION}.txt"
```

---

### 场景7: 自动生成（不指定任何路径参数）

**适用于**: 快速测试，让系统自动管理文件名

```bash
python main.py \
    --position_encoding relative \
    --norm_type rmsnorm \
    --data_size 10k \
    --epoch_num 5
```

**自动生成的文件**:
```
experiment/
├── model_10k_pos-relative_norm-rmsnorm.pth
├── train_10k_pos-relative_norm-rmsnorm.log
└── output_10k_pos-relative_norm-rmsnorm.txt
```

---

## 💡 实用技巧

### 技巧1: 使用时间戳避免覆盖

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python main.py \
    --exp_name "experiment_${TIMESTAMP}"
```

### 技巧2: 将多个实验保存到同一目录

```bash
EXP_DIR="./my_experiment_batch"
mkdir -p ${EXP_DIR}

# 实验1
python main.py \
    --position_encoding absolute \
    --model_path "${EXP_DIR}/model_absolute.pth" \
    --log_path "${EXP_DIR}/train_absolute.log"

# 实验2
python main.py \
    --position_encoding relative \
    --model_path "${EXP_DIR}/model_relative.pth" \
    --log_path "${EXP_DIR}/train_relative.log"
```

### 技巧3: 使用相对路径和绝对路径

```bash
# 相对路径（相对于当前工作目录）
python main.py --model_path "./models/model.pth"

# 绝对路径（完整路径）
python main.py --model_path "/home/user/nmt/models/model.pth"
```

### 技巧4: 在远程服务器上保存到共享目录

```bash
# 保存到共享存储
python main.py \
    --model_path "/mnt/shared_storage/models/model.pth" \
    --log_path "/mnt/shared_storage/logs/train.log"
```

---

## 🔍 路径优先级规则

代码会按以下优先级处理路径：

1. **最高优先级**: 命令行指定的完整路径
   - `--model_path`, `--log_path`, `--output_path`

2. **中等优先级**: 命令行指定的实验名称
   - `--exp_name`

3. **最低优先级**: 根据架构配置自动生成
   - 基于 `--position_encoding` 和 `--norm_type`

---

## 📊 对比总结

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| `--exp_name` | 简单，统一管理 | 固定在experiment/目录 | 标准实验 |
| `--model_path` 等 | 完全自定义，灵活 | 需要手动管理路径 | 特殊需求 |
| 自动生成 | 零配置，自动命名 | 名称较长 | 快速测试 |

---

## ⚠️ 注意事项

1. **目录必须存在**: 使用自定义路径前，确保目录已创建
   ```bash
   mkdir -p my_models my_logs my_results
   ```

2. **文件会被覆盖**: 如果文件已存在，训练会覆盖它
   - 使用时间戳避免覆盖
   - 或手动备份重要文件

3. **路径格式**:
   - Linux/Mac: 使用 `/` (斜杠)
   - Windows: 使用 `\\` 或 `/` 都可以

4. **权限问题**: 确保对目标目录有写权限

---

## 🎓 完整示例脚本

创建一个组织良好的实验目录结构：

```bash
#!/bin/bash
# create_experiment.sh

# 设置实验参数
EXPERIMENT_NAME="ablation_study_20251218"
BASE_DIR="./experiments/${EXPERIMENT_NAME}"

# 创建目录结构
mkdir -p ${BASE_DIR}/{models,logs,results,configs}

# 保存实验配置
cat > ${BASE_DIR}/configs/experiment_config.txt <<EOF
Experiment: ${EXPERIMENT_NAME}
Date: $(date)
Position Encodings: absolute, relative
Normalizations: layernorm, rmsnorm
Dataset: 10k
Epochs: 40
EOF

# 运行实验
VARIANTS=("absolute_layernorm" "relative_layernorm" "absolute_rmsnorm" "relative_rmsnorm")
POS_TYPES=("absolute" "relative" "absolute" "relative")
NORM_TYPES=("layernorm" "layernorm" "rmsnorm" "rmsnorm")

for i in ${!VARIANTS[@]}; do
    VARIANT=${VARIANTS[$i]}
    POS=${POS_TYPES[$i]}
    NORM=${NORM_TYPES[$i]}

    echo "Running experiment: ${VARIANT}"

    python main.py \
        --position_encoding ${POS} \
        --norm_type ${NORM} \
        --data_size 10k \
        --epoch_num 40 \
        --model_path "${BASE_DIR}/models/${VARIANT}.pth" \
        --log_path "${BASE_DIR}/logs/${VARIANT}.log" \
        --output_path "${BASE_DIR}/results/${VARIANT}.txt"
done

echo "All experiments completed!"
echo "Results saved in: ${BASE_DIR}"
```

使用方法：
```bash
chmod +x create_experiment.sh
./create_experiment.sh
```

---

## ✅ 验证路径设置

训练开始时会打印配置信息，包括路径：

```
============================================================
🚀 Training Configuration
============================================================
...
Model will be saved to: ./my_models/transformer_v1.pth
============================================================
```

确认路径正确后，训练会继续进行。
