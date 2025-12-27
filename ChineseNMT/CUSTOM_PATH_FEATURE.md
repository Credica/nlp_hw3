# 自定义路径功能说明

## 🎯 新增功能

现在支持通过命令行参数自定义模型、日志和输出文件的保存路径和名称！

---

## 📋 新增的命令行参数

```bash
--model_path PATH     # 自定义模型保存路径（完整路径和文件名）
--log_path PATH       # 自定义训练日志路径（完整路径和文件名）
--output_path PATH    # 自定义翻译输出路径（完整路径和文件名）
```

---

## 🚀 使用示例

### 示例1: 完全自定义路径

```bash
python main.py \
    --position_encoding relative \
    --data_size 10k \
    --epoch_num 40 \
    --model_path "./my_models/transformer_v1.pth" \
    --log_path "./my_logs/training_v1.log" \
    --output_path "./my_results/output_v1.txt"
```

**说明**: 所有文件都保存到指定的位置，需要先创建目录：
```bash
mkdir -p my_models my_logs my_results
```

---

### 示例2: 只自定义模型路径

```bash
python main.py \
    --position_encoding relative \
    --exp_name "test_v1" \
    --model_path "./saved_models/best_model.pth"
```

**说明**:
- 模型保存到: `./saved_models/best_model.pth` (自定义)
- 日志保存到: `./experiment/train_test_v1.log` (使用exp_name)
- 输出保存到: `./experiment/output_test_v1.txt` (使用exp_name)

---

### 示例3: 使用时间戳命名

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python main.py \
    --position_encoding relative \
    --model_path "./models/model_${TIMESTAMP}.pth" \
    --log_path "./logs/train_${TIMESTAMP}.log"
```

**生成的文件**:
```
models/model_20251218_143022.pth
logs/train_20251218_143022.log
experiment/output_10k_pos-relative_norm-layernorm.txt  # 自动生成
```

---

### 示例4: 按实验组织

```bash
EXP_DIR="./experiments/relative_position_test"
mkdir -p ${EXP_DIR}

python main.py \
    --position_encoding relative \
    --data_size 10k \
    --model_path "${EXP_DIR}/model.pth" \
    --log_path "${EXP_DIR}/train.log" \
    --output_path "${EXP_DIR}/output.txt"
```

**生成的目录结构**:
```
experiments/relative_position_test/
├── model.pth
├── train.log
└── output.txt
```

---

### 示例5: 传统方式（使用exp_name）

```bash
python main.py \
    --position_encoding relative \
    --exp_name "my_experiment"
```

**生成的文件**:
```
experiment/
├── model_my_experiment.pth
├── train_my_experiment.log
└── output_my_experiment.txt
```

---

## 📊 对比：三种命名方式

| 方式 | 命令 | 结果 |
|------|------|------|
| **1. 自定义路径** | `--model_path "./models/v1.pth"` | 完全控制路径和文件名 |
| **2. 实验名称** | `--exp_name "test"` | 自动在experiment/下生成 |
| **3. 自动生成** | 不指定任何参数 | 根据配置自动命名 |

---

## 🔍 路径优先级

代码按以下优先级处理路径：

1. **最高**: `--model_path`, `--log_path`, `--output_path`（完全自定义）
2. **中等**: `--exp_name`（自动在experiment/目录下生成）
3. **最低**: 根据架构配置自动生成（如: `model_10k_pos-relative_norm-rmsnorm.pth`）

---

## ✅ 实用场景

### 场景1: 保存到专门的模型目录

```bash
mkdir -p best_models
python main.py \
    --data_size 100k \
    --epoch_num 40 \
    --model_path "./best_models/transformer_100k_final.pth"
```

### 场景2: 将所有实验文件保存到一个目录

```bash
EXP_NAME="ablation_$(date +%Y%m%d)"
EXP_DIR="./all_experiments/${EXP_NAME}"
mkdir -p ${EXP_DIR}

for pos in absolute relative; do
    for norm in layernorm rmsnorm; do
        python main.py \
            --position_encoding ${pos} \
            --norm_type ${norm} \
            --data_size 10k \
            --model_path "${EXP_DIR}/model_${pos}_${norm}.pth" \
            --log_path "${EXP_DIR}/train_${pos}_${norm}.log" \
            --output_path "${EXP_DIR}/output_${pos}_${norm}.txt"
    done
done
```

### 场景3: 保存到共享存储

```bash
python main.py \
    --data_size 100k \
    --model_path "/mnt/shared/models/transformer_v1.pth" \
    --log_path "/mnt/shared/logs/train.log"
```

---

## 📚 查看完整示例

详细的使用示例请查看: **`CUSTOM_PATH_EXAMPLES.md`**

该文档包含：
- 7种不同的使用场景
- 完整的shell脚本示例
- 路径命名技巧
- 常见问题解决方案

---

## ⚠️ 注意事项

1. **目录必须存在**: 使用自定义路径前，确保目录已创建
   ```bash
   mkdir -p my_models my_logs my_results
   ```

2. **文件会被覆盖**: 如果文件已存在，新的训练会覆盖它
   - 建议使用时间戳避免覆盖
   - 或手动备份重要文件

3. **权限检查**: 确保对目标目录有写权限

4. **路径格式**:
   - 相对路径: `./my_models/model.pth`
   - 绝对路径: `/home/user/models/model.pth`
   - Windows路径: 使用 `/` 或 `\\` 都可以

---

## 🎉 快速测试

验证新功能是否正常工作：

```bash
# 创建测试目录
mkdir -p test_output

# 运行一个快速测试（5个epoch）
python main.py \
    --position_encoding relative \
    --data_size 10k \
    --epoch_num 5 \
    --model_path "./test_output/test_model.pth" \
    --log_path "./test_output/test_train.log" \
    --output_path "./test_output/test_output.txt"

# 检查文件是否生成
ls -lh test_output/
```

---

## 💡 推荐使用方式

### 日常实验（推荐）
使用 `--exp_name`，简单且不易出错：
```bash
python main.py --exp_name "my_test"
```

### 正式实验（推荐）
使用完全自定义路径，便于管理：
```bash
EXP_DIR="./experiments/$(date +%Y%m%d)_final_test"
mkdir -p ${EXP_DIR}
python main.py \
    --model_path "${EXP_DIR}/model.pth" \
    --log_path "${EXP_DIR}/train.log"
```

### 快速测试
让系统自动生成，无需配置：
```bash
python main.py --position_encoding relative
```

---

**现在你可以完全掌控文件的保存位置和命名方式！** 🎉
