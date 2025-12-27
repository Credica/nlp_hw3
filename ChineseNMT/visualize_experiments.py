"""
Transformer实验结果可视化脚本
用于分析和对比不同配置下的Transformer模型性能
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def parse_log_file(log_path):
    """解析训练日志文件，提取训练过程数据"""
    epochs = []
    train_losses = []
    dev_losses = []
    bleu_scores = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取epoch训练损失
            if match := re.search(r'Epoch: (\d+), loss: ([\d.]+)', line):
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                epochs.append(epoch)
                train_losses.append(train_loss)

            # 提取验证损失和BLEU分数
            elif match := re.search(r'Epoch: \d+, Dev loss: ([\d.]+), Bleu Score: ([\d.]+)', line):
                dev_loss = float(match.group(1))
                bleu_score = float(match.group(2))
                dev_losses.append(dev_loss)
                bleu_scores.append(bleu_score)

            # 提取最终测试集BLEU分数
            elif match := re.search(r'Final Test Set BLEU Score: ([\d.]+)', line):
                test_bleu = float(match.group(1))

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'dev_losses': dev_losses,
        'bleu_scores': bleu_scores,
        'test_bleu': test_bleu if 'test_bleu' in locals() else None
    }


def parse_experiment_name(exp_name):
    """从实验名称中提取配置信息"""
    # 格式: 100k_abs_ln_adam_nosmooth
    parts = exp_name.split('_')

    return {
        'dataset': parts[0],
        'position_encoding': 'Absolute' if parts[1] == 'abs' else 'Relative',
        'normalization': 'LayerNorm' if parts[2] == 'ln' else 'RMSNorm',
        'optimizer': 'Adam' if parts[3] == 'adam' else 'Noam',
        'label_smoothing': 'Smooth' if parts[4] == 'smooth' else 'No Smooth',
        'config_name': f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
    }


def collect_all_results(base_dir):
    """收集所有实验的结果"""
    results = []

    for exp_dir in Path(base_dir).glob('100k_*'):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        log_file = exp_dir / f"train_{exp_name}.log"

        if not log_file.exists():
            print(f"Warning: Log file not found for {exp_name}")
            continue

        # 解析日志文件
        log_data = parse_log_file(log_file)

        # 解析实验配置
        config = parse_experiment_name(exp_name)

        # 合并结果
        result = {**config, **log_data, 'exp_name': exp_name}
        results.append(result)

    return results


def plot_final_bleu_comparison(results, save_dir):
    """绘制最终BLEU分数对比图"""
    # 准备数据
    data = []
    for r in results:
        if r['test_bleu'] is not None:
            data.append({
                'Experiment': r['config_name'],
                'Position': r['position_encoding'],
                'Norm': r['normalization'],
                'Optimizer': r['optimizer'],
                'Smoothing': r['label_smoothing'],
                'BLEU': r['test_bleu']
            })

    df = pd.DataFrame(data)
    df = df.sort_values('BLEU', ascending=False)

    # 绘制条形图
    fig, ax = plt.subplots(figsize=(16, 8))

    # 根据配置着色
    colors = []
    for _, row in df.iterrows():
        if row['Optimizer'] == 'Noam':
            colors.append('steelblue')
        else:
            colors.append('coral')

    bars = ax.barh(df['Experiment'], df['BLEU'], color=colors, alpha=0.8)

    # 在条形上标注具体数值
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['BLEU'] + 0.05, i, f"{row['BLEU']:.2f}",
                va='center', fontsize=9)

    ax.set_xlabel('Test BLEU Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Configurations: Test BLEU Score Comparison\n(Blue: Noam Optimizer, Orange: Adam Optimizer)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # 添加图例 - 移到右上角
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Noam Optimizer'),
        Patch(facecolor='coral', alpha=0.8, label='Adam Optimizer')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/bleu_comparison_all.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: bleu_comparison_all.png")


def plot_training_curves(results, save_dir, max_experiments=8):
    """绘制训练曲线对比"""
    # 选择表现最好的几个实验
    sorted_results = sorted(results, key=lambda x: x['test_bleu'] if x['test_bleu'] else 0, reverse=True)
    top_results = sorted_results[:max_experiments]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 训练损失曲线
    ax = axes[0]
    for r in top_results:
        ax.plot(r['epochs'], r['train_losses'], label=r['config_name'], linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Curves (Top 8 Configurations)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # 2. 验证BLEU分数曲线
    ax = axes[1]
    for r in top_results:
        ax.plot(r['epochs'], r['bleu_scores'], label=r['config_name'], linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation BLEU Score', fontweight='bold')
    ax.set_title('Validation BLEU Score Curves (Top 8 Configurations)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: training_curves.png")


def plot_ablation_study(results, save_dir):
    """绘制消融研究分析 - 使用单个表格展示所有统计结果"""
    data = []
    for r in results:
        if r['test_bleu'] is not None:
            data.append({
                'Position Encoding': r['position_encoding'],
                'Normalization': r['normalization'],
                'Optimizer': r['optimizer'],
                'Label Smoothing': r['label_smoothing'],
                'BLEU': r['test_bleu']
            })

    df = pd.DataFrame(data)

    # 准备汇总表格数据
    table_data = []
    table_data.append(['Factor', 'Category', 'Count', 'Mean', 'Std', 'Min', 'Max'])

    factors = ['Position Encoding', 'Normalization', 'Optimizer', 'Label Smoothing']
    factor_rows = {}  # 记录每个因素的起始行和行数

    row_idx = 1
    for factor in factors:
        stats = df.groupby(factor)['BLEU'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        factor_rows[factor] = (row_idx, len(stats))

        for i, category in enumerate(stats.index):
            row = [
                factor if i == 0 else '',  # 只在第一行显示因素名称
                category,
                int(stats.loc[category, 'count']),
                f"{stats.loc[category, 'mean']:.2f}",
                f"{stats.loc[category, 'std']:.2f}",
                f"{stats.loc[category, 'min']:.2f}",
                f"{stats.loc[category, 'max']:.2f}"
            ]
            table_data.append(row)
            row_idx += 1

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # 创建表格
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.22, 0.20, 0.10, 0.12, 0.12, 0.12, 0.12])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)

    # 设置表头样式
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_height(0.08)

    # 设置数据行样式
    row_idx = 1
    for factor in factors:
        stats = df.groupby(factor)['BLEU'].agg(['mean']).round(2)
        max_mean = stats['mean'].max()
        start_row, num_rows = factor_rows[factor]

        for i, category in enumerate(stats.index):
            for j in range(7):
                cell = table[(row_idx, j)]

                # 交替背景色
                if row_idx % 2 == 0:
                    cell.set_facecolor('#F2F2F2')
                else:
                    cell.set_facecolor('#FFFFFF')

                # 高亮显示最优均值
                if j == 3:  # Mean列
                    mean_val = float(table_data[row_idx][3])
                    if abs(mean_val - max_mean) < 0.01:
                        cell.set_facecolor('#92D050')
                        cell.set_text_props(weight='bold')

                # 处理Factor列 - 模拟合并单元格
                if j == 0:
                    if i == 0:
                        # 第一行：显示因素名称，居中对齐
                        cell.set_text_props(weight='bold', fontsize=11, va='center')
                        # 移除下边框（除了最后一行）
                        if i < num_rows - 1:
                            cell.visible_edges = 'LTR'  # 左、上、右边框
                    else:
                        # 后续行：隐藏文本，移除上边框
                        cell.get_text().set_text('')
                        if i < num_rows - 1:
                            cell.visible_edges = 'LR'  # 只保留左右边框
                        else:
                            cell.visible_edges = 'LBR'  # 左、下、右边框

            row_idx += 1

        # 在每个因素组之间添加粗分隔线
        if factor != factors[-1]:
            for j in range(7):
                cell = table[(row_idx - 1, j)]
                cell.set_linewidth(2.5)

    plt.title('Ablation Study: Statistical Summary of All Factors\n(Green highlight: Best average performance within each factor)',
              fontsize=14, fontweight='bold', pad=15)  # 减小pad从30到15
    plt.savefig(f"{save_dir}/ablation_study.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: ablation_study.png (single table format)")

    # 同时保存CSV格式
    with open(f"{save_dir}/ablation_study_stats.csv", 'w', encoding='utf-8-sig') as f:
        f.write("Factor,Category,Count,Mean,Std,Min,Max\n")
        for row in table_data[1:]:
            f.write(','.join(map(str, row)) + '\n')

    print(f"✓ Saved: ablation_study_stats.csv")


def plot_interaction_heatmap(results, save_dir):
    """绘制配置交互效应热力图"""
    data = []
    for r in results:
        if r['test_bleu'] is not None:
            data.append({
                'Position': r['position_encoding'][:3],  # Abs/Rel
                'Norm': r['normalization'][:2],          # La/RM
                'Optimizer': r['optimizer'],
                'Smoothing': r['label_smoothing'][:2],   # No/Sm
                'BLEU': r['test_bleu']
            })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 位置编码 vs 优化器
    ax = axes[0]
    pivot1 = df.pivot_table(values='BLEU', index='Position', columns='Optimizer', aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Average BLEU'}, vmin=df['BLEU'].min(), vmax=df['BLEU'].max())
    ax.set_title('Position Encoding vs Optimizer', fontweight='bold', fontsize=12)
    ax.set_xlabel('Optimizer', fontweight='bold')
    ax.set_ylabel('Position Encoding', fontweight='bold')

    # 2. 归一化 vs 标签平滑
    ax = axes[1]
    pivot2 = df.pivot_table(values='BLEU', index='Norm', columns='Smoothing', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'Average BLEU'}, vmin=df['BLEU'].min(), vmax=df['BLEU'].max())
    ax.set_title('Normalization vs Label Smoothing', fontweight='bold', fontsize=12)
    ax.set_xlabel('Label Smoothing', fontweight='bold')
    ax.set_ylabel('Normalization', fontweight='bold')

    plt.suptitle('Configuration Interaction Effects on BLEU Score',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/interaction_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: interaction_heatmap.png")


def generate_summary_table(results, save_dir):
    """生成实验结果汇总表"""
    data = []
    for r in results:
        if r['test_bleu'] is not None:
            best_epoch = r['epochs'][np.argmax(r['bleu_scores'])]
            best_val_bleu = max(r['bleu_scores'])

            data.append({
                'Configuration': r['config_name'],
                'Position Encoding': r['position_encoding'],
                'Normalization': r['normalization'],
                'Optimizer': r['optimizer'],
                'Label Smoothing': r['label_smoothing'],
                'Best Epoch': best_epoch,
                'Best Val BLEU': f"{best_val_bleu:.2f}",
                'Test BLEU': f"{r['test_bleu']:.2f}",
                'Final Train Loss': f"{r['train_losses'][-1]:.4f}",
                'Final Val Loss': f"{r['dev_losses'][-1]:.4f}"
            })

    df = pd.DataFrame(data)
    df = df.sort_values('Test BLEU', ascending=False)

    # 保存为CSV
    df.to_csv(f"{save_dir}/experiment_summary.csv", index=False, encoding='utf-8-sig')

    # 保存为Markdown格式
    with open(f"{save_dir}/experiment_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Transformer Experiment Results Summary\n\n")
        f.write("## Overall Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Top 5 Configurations\n\n")
        f.write(df.head(5).to_markdown(index=False))

        # 添加统计分析
        f.write("\n\n## Statistical Analysis\n\n")
        f.write(f"- **Best BLEU Score**: {df['Test BLEU'].max()} ({df.iloc[0]['Configuration']})\n")
        f.write(f"- **Worst BLEU Score**: {df['Test BLEU'].min()}\n")
        f.write(f"- **Average BLEU Score**: {pd.to_numeric(df['Test BLEU']).mean():.2f}\n")
        f.write(f"- **Standard Deviation**: {pd.to_numeric(df['Test BLEU']).std():.2f}\n")

    print(f"✓ Saved: experiment_summary.csv and experiment_summary.md")

    return df


def main():
    """主函数"""
    print("=" * 60)
    print("Transformer Experiment Visualization Script")
    print("=" * 60)

    # 设置路径
    base_dir = "./experiment"
    save_dir = "./figures"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📁 Reading experiments from: {base_dir}")
    print(f"📊 Saving visualizations to: {save_dir}\n")

    # 收集所有实验结果
    print("Collecting experiment results...")
    results = collect_all_results(base_dir)
    print(f"✓ Found {len(results)} experiments\n")

    if len(results) == 0:
        print("❌ No experiment results found!")
        return

    # 生成汇总表
    print("Generating summary table...")
    summary_df = generate_summary_table(results, save_dir)
    print()

    # 绘制可视化图表
    print("Generating visualizations...")
    print("-" * 60)

    plot_final_bleu_comparison(results, save_dir)
    plot_training_curves(results, save_dir)
    plot_ablation_study(results, save_dir)
    plot_interaction_heatmap(results, save_dir)

    print("-" * 60)
    print("\n✅ All visualizations completed!")
    print(f"\n📊 Generated files in {save_dir}/:")
    print("  - bleu_comparison_all.png")
    print("  - training_curves.png")
    print("  - ablation_study.png")
    print("  - interaction_heatmap.png")
    print("  - experiment_summary.csv")
    print("  - experiment_summary.md")

    # 显示Top 5结果
    print("\n" + "=" * 60)
    print("Top 5 Configurations by Test BLEU Score")
    print("=" * 60)
    print(summary_df.head(5)[['Configuration', 'Test BLEU', 'Best Val BLEU']].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
