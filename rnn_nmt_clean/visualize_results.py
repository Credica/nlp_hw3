#!/usr/bin/env python3
"""
NMT实验结果可视化脚本
可视化不同attention机制和teacher forcing ratio的训练结果
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# 数据目录
CHECKPOINT_DIR = '/mnt/afs/250010024/nlp_hw3/rnn_nmt_clean/checkpoints_100k'
OUTPUT_DIR = '/mnt/afs/250010024/nlp_hw3/rnn_nmt_clean/figures'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实验配置
experiments = {
    'additive_tf0': {
        'name': 'Additive (TF=0%)',
        'attention': 'additive',
        'tf_ratio': 0.0,
        'color': '#1f77b4',
        'linestyle': '-'
    },
    'additive_tf50': {
        'name': 'Additive (TF=50%)',
        'attention': 'additive',
        'tf_ratio': 0.5,
        'color': '#2ca02c',
        'linestyle': '-'
    },
    'additive_tf100': {
        'name': 'Additive (TF=100%)',
        'attention': 'additive',
        'tf_ratio': 1.0,
        'color': '#d62728',
        'linestyle': '-'
    },
    'dot_tf50': {
        'name': 'Dot Product (TF=50%)',
        'attention': 'dot',
        'tf_ratio': 0.5,
        'color': '#9467bd',
        'linestyle': '--'
    },
    'multiplicative_tf50': {
        'name': 'Multiplicative (TF=50%)',
        'attention': 'multiplicative',
        'tf_ratio': 0.5,
        'color': '#ff7f0e',
        'linestyle': '-.'
    }
}

# BLEU分数数据
bleu_scores = {
    'additive_tf0': {'greedy': 7.54, 'beam3': 1.39, 'beam5': 1.38, 'beam10': 7.54, 'best_loss': 6.1181},
    'additive_tf50': {'greedy': 8.10, 'beam3': 7.17, 'beam5': 9.35, 'beam10': 8.31, 'best_loss': 6.3376},
    'additive_tf100': {'greedy': 9.22, 'beam3': 6.04, 'beam5': 9.31, 'beam10': 9.00, 'best_loss': 9.2474},
    'dot_tf50': {'greedy': 7.54, 'beam3': 2.68, 'beam5': 6.47, 'beam10': 7.76, 'best_loss': 6.4089},
    'multiplicative_tf50': {'greedy': 9.92, 'beam3': 3.13, 'beam5': 10.05, 'beam10': 9.92, 'best_loss': 6.5639}
}


def load_training_log(exp_name):
    """加载训练日志"""
    json_path = os.path.join(CHECKPOINT_DIR, f'results_100k_{exp_name}_training_log.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_loss_curves():
    """绘制训练和验证Loss曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name, config in experiments.items():
        data = load_training_log(exp_name)
        epochs = range(1, len(data['train_losses']) + 1)

        # 训练Loss
        axes[0].plot(epochs, data['train_losses'],
                     label=config['name'], color=config['color'],
                     linestyle=config['linestyle'], linewidth=2, marker='o', markersize=4)

        # 验证Loss
        axes[1].plot(epochs, data['valid_losses'],
                     label=config['name'], color=config['color'],
                     linestyle=config['linestyle'], linewidth=2, marker='o', markersize=4)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.pdf'), bbox_inches='tight')
    print(f"Saved: loss_curves.png/pdf")
    plt.close()


def plot_perplexity_curves():
    """绘制训练和验证Perplexity曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name, config in experiments.items():
        data = load_training_log(exp_name)
        epochs = range(1, len(data['train_ppls']) + 1)

        # 训练PPL
        axes[0].plot(epochs, data['train_ppls'],
                     label=config['name'], color=config['color'],
                     linestyle=config['linestyle'], linewidth=2, marker='o', markersize=4)

        # 验证PPL (使用对数刻度，因为PPL差异很大)
        axes[1].plot(epochs, data['valid_ppls'],
                     label=config['name'], color=config['color'],
                     linestyle=config['linestyle'], linewidth=2, marker='o', markersize=4)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Perplexity', fontsize=12)
    axes[0].set_title('Training Perplexity Curves', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Perplexity', fontsize=12)
    axes[1].set_title('Validation Perplexity Curves', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_yscale('log')  # 对数刻度
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'perplexity_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'perplexity_curves.pdf'), bbox_inches='tight')
    print(f"Saved: perplexity_curves.png/pdf")
    plt.close()


def plot_bleu_comparison():
    """绘制不同解码策略的BLEU分数对比"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(experiments))
    width = 0.2

    beam_sizes = ['greedy', 'beam3', 'beam5', 'beam10']
    beam_labels = ['Greedy (Beam-1)', 'Beam-3', 'Beam-5', 'Beam-10']
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

    for i, (beam, label, color) in enumerate(zip(beam_sizes, beam_labels, colors)):
        scores = [bleu_scores[exp][beam] for exp in experiments.keys()]
        bars = ax.bar(x + i * width, scores, width, label=label, color=color, edgecolor='black', linewidth=0.5)

        # 在柱子上显示数值
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Experiment Configuration', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title('BLEU-4 Scores: Different Decoding Strategies', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([config['name'] for config in experiments.values()], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bleu_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'bleu_comparison.pdf'), bbox_inches='tight')
    print(f"Saved: bleu_comparison.png/pdf")
    plt.close()


def plot_attention_comparison():
    """绘制不同Attention机制的对比（固定TF=50%）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 只选择TF=50%的实验
    tf50_exps = ['additive_tf50', 'dot_tf50', 'multiplicative_tf50']

    # Loss对比
    for exp_name in tf50_exps:
        config = experiments[exp_name]
        data = load_training_log(exp_name)
        epochs = range(1, len(data['valid_losses']) + 1)
        axes[0].plot(epochs, data['valid_losses'],
                     label=config['name'].replace(' (TF=50%)', ''),
                     color=config['color'], linestyle=config['linestyle'],
                     linewidth=2.5, marker='o', markersize=5)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss: Attention Mechanisms (TF=50%)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # BLEU对比柱状图
    attention_types = ['Additive', 'Dot Product', 'Multiplicative']
    colors = ['#2ca02c', '#9467bd', '#ff7f0e']

    x = np.arange(len(attention_types))
    width = 0.25

    greedy_scores = [bleu_scores[exp]['greedy'] for exp in tf50_exps]
    beam5_scores = [bleu_scores[exp]['beam5'] for exp in tf50_exps]
    beam10_scores = [bleu_scores[exp]['beam10'] for exp in tf50_exps]

    bars1 = axes[1].bar(x - width, greedy_scores, width, label='Greedy', color='#4e79a7', edgecolor='black')
    bars2 = axes[1].bar(x, beam5_scores, width, label='Beam-5', color='#e15759', edgecolor='black')
    bars3 = axes[1].bar(x + width, beam10_scores, width, label='Beam-10', color='#76b7b2', edgecolor='black')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    axes[1].set_xlabel('Attention Type', fontsize=12)
    axes[1].set_ylabel('BLEU-4 Score', fontsize=12)
    axes[1].set_title('BLEU-4: Attention Mechanisms (TF=50%)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(attention_types)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_comparison.pdf'), bbox_inches='tight')
    print(f"Saved: attention_comparison.png/pdf")
    plt.close()


def plot_teacher_forcing_comparison():
    """绘制不同Teacher Forcing Ratio的对比（Additive Attention）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 只选择Additive Attention的实验
    additive_exps = ['additive_tf0', 'additive_tf50', 'additive_tf100']
    tf_labels = ['TF=0%', 'TF=50%', 'TF=100%']

    # Loss对比
    for exp_name in additive_exps:
        config = experiments[exp_name]
        data = load_training_log(exp_name)
        epochs = range(1, len(data['valid_losses']) + 1)
        axes[0].plot(epochs, data['valid_losses'],
                     label=config['name'].replace('Additive ', ''),
                     color=config['color'], linestyle='-',
                     linewidth=2.5, marker='o', markersize=5)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss: Teacher Forcing Ratio (Additive)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # BLEU对比柱状图
    x = np.arange(len(tf_labels))
    width = 0.25

    greedy_scores = [bleu_scores[exp]['greedy'] for exp in additive_exps]
    beam5_scores = [bleu_scores[exp]['beam5'] for exp in additive_exps]
    beam10_scores = [bleu_scores[exp]['beam10'] for exp in additive_exps]

    bars1 = axes[1].bar(x - width, greedy_scores, width, label='Greedy', color='#4e79a7', edgecolor='black')
    bars2 = axes[1].bar(x, beam5_scores, width, label='Beam-5', color='#e15759', edgecolor='black')
    bars3 = axes[1].bar(x + width, beam10_scores, width, label='Beam-10', color='#76b7b2', edgecolor='black')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    axes[1].set_xlabel('Teacher Forcing Ratio', fontsize=12)
    axes[1].set_ylabel('BLEU-4 Score', fontsize=12)
    axes[1].set_title('BLEU-4: Teacher Forcing Ratio (Additive)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tf_labels)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'teacher_forcing_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'teacher_forcing_comparison.pdf'), bbox_inches='tight')
    print(f"Saved: teacher_forcing_comparison.png/pdf")
    plt.close()


def plot_summary_table():
    """绘制实验结果汇总表格"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # 表格数据
    columns = ['Experiment', 'Attention', 'TF Ratio', 'Best Val Loss',
               'Greedy', 'Beam-3', 'Beam-5', 'Beam-10', 'Best BLEU']

    data = []
    for exp_name, config in experiments.items():
        bleu = bleu_scores[exp_name]
        best_bleu = max(bleu['greedy'], bleu['beam3'], bleu['beam5'], bleu['beam10'])
        data.append([
            config['name'],
            config['attention'].capitalize(),
            f"{config['tf_ratio']*100:.0f}%",
            f"{bleu['best_loss']:.4f}",
            f"{bleu['greedy']:.2f}",
            f"{bleu['beam3']:.2f}",
            f"{bleu['beam5']:.2f}",
            f"{bleu['beam10']:.2f}",
            f"{best_bleu:.2f}"
        ])

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 设置交替行颜色
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title('NMT Experiment Results Summary (100k Dataset)', fontsize=14, fontweight='bold', y=0.95)

    plt.savefig(os.path.join(OUTPUT_DIR, 'results_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_summary_table.pdf'), bbox_inches='tight')
    print(f"Saved: results_summary_table.png/pdf")
    plt.close()


def plot_best_bleu_bar():
    """绘制各实验最佳BLEU分数对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_names = list(experiments.keys())
    best_bleus = []
    colors = []

    for exp_name in exp_names:
        bleu = bleu_scores[exp_name]
        best_bleu = max(bleu['greedy'], bleu['beam3'], bleu['beam5'], bleu['beam10'])
        best_bleus.append(best_bleu)
        colors.append(experiments[exp_name]['color'])

    x = np.arange(len(exp_names))
    bars = ax.bar(x, best_bleus, color=colors, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, score in zip(bars, best_bleus):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Experiment Configuration', fontsize=12)
    ax.set_ylabel('Best BLEU-4 Score', fontsize=12)
    ax.set_title('Best BLEU-4 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([experiments[exp]['name'] for exp in exp_names], rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 12)

    # 标注最佳实验
    best_idx = np.argmax(best_bleus)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'best_bleu_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'best_bleu_comparison.pdf'), bbox_inches='tight')
    print(f"Saved: best_bleu_comparison.png/pdf")
    plt.close()


def plot_all_in_one():
    """绘制综合图（2x2布局）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 验证Loss曲线
    ax = axes[0, 0]
    for exp_name, config in experiments.items():
        data = load_training_log(exp_name)
        epochs = range(1, len(data['valid_losses']) + 1)
        ax.plot(epochs, data['valid_losses'],
                label=config['name'], color=config['color'],
                linestyle=config['linestyle'], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('(a) Validation Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 验证Perplexity曲线
    ax = axes[0, 1]
    for exp_name, config in experiments.items():
        data = load_training_log(exp_name)
        epochs = range(1, len(data['valid_ppls']) + 1)
        ax.plot(epochs, data['valid_ppls'],
                label=config['name'], color=config['color'],
                linestyle=config['linestyle'], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Perplexity (log scale)', fontsize=11)
    ax.set_title('(b) Validation Perplexity Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. BLEU分数对比（按实验）
    ax = axes[1, 0]
    x = np.arange(len(experiments))
    width = 0.2
    beam_sizes = ['greedy', 'beam3', 'beam5', 'beam10']
    beam_labels = ['Greedy', 'Beam-3', 'Beam-5', 'Beam-10']
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

    for i, (beam, label, color) in enumerate(zip(beam_sizes, beam_labels, colors)):
        scores = [bleu_scores[exp][beam] for exp in experiments.keys()]
        ax.bar(x + i * width, scores, width, label=label, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel('BLEU-4 Score', fontsize=11)
    ax.set_title('(c) BLEU-4 Scores by Decoding Strategy', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([config['name'].split(' ')[0] + '\n' + ' '.join(config['name'].split(' ')[1:])
                        for config in experiments.values()], fontsize=8)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 12)

    # 4. 最佳BLEU对比
    ax = axes[1, 1]
    exp_names = list(experiments.keys())
    best_bleus = []
    bar_colors = []

    for exp_name in exp_names:
        bleu = bleu_scores[exp_name]
        best_bleu = max(bleu['greedy'], bleu['beam3'], bleu['beam5'], bleu['beam10'])
        best_bleus.append(best_bleu)
        bar_colors.append(experiments[exp_name]['color'])

    bars = ax.bar(range(len(exp_names)), best_bleus, color=bar_colors, edgecolor='black', linewidth=1)

    for bar, score in zip(bars, best_bleus):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Experiment', fontsize=11)
    ax.set_ylabel('Best BLEU-4 Score', fontsize=11)
    ax.set_title('(d) Best BLEU-4 Score Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels([config['name'].split(' ')[0] + '\n' + ' '.join(config['name'].split(' ')[1:])
                        for config in experiments.values()], fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 12)

    # 标注最佳
    best_idx = np.argmax(best_bleus)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    plt.suptitle('NMT Experiment Results Summary (100k Dataset)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_results_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_results_summary.pdf'), bbox_inches='tight')
    print(f"Saved: all_results_summary.png/pdf")
    plt.close()


def main():
    print("=" * 60)
    print("NMT Experiment Visualization")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}\n")

    print("Generating plots...")

    # 生成各种可视化图表
    plot_loss_curves()
    plot_perplexity_curves()
    plot_bleu_comparison()
    plot_attention_comparison()
    plot_teacher_forcing_comparison()
    plot_summary_table()
    plot_best_bleu_bar()
    plot_all_in_one()

    print("\n" + "=" * 60)
    print("All visualizations saved successfully!")
    print("=" * 60)

    # 打印实验结果摘要
    print("\n[Experiment Results Summary]")
    print("-" * 60)

    best_exp = None
    best_score = 0

    for exp_name, config in experiments.items():
        bleu = bleu_scores[exp_name]
        best_bleu = max(bleu['greedy'], bleu['beam3'], bleu['beam5'], bleu['beam10'])
        if best_bleu > best_score:
            best_score = best_bleu
            best_exp = exp_name
        print(f"{config['name']:25s}: Best BLEU = {best_bleu:.2f} (Val Loss = {bleu['best_loss']:.4f})")

    print("-" * 60)
    print(f"Best Experiment: {experiments[best_exp]['name']} (BLEU-4 = {best_score:.2f})")


if __name__ == '__main__':
    main()
