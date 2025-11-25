#!/usr/bin/env python3
"""
从results.csv生成训练可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def generate_training_plots(results_dir):
    """生成训练曲线图"""
    results_dir = Path(results_dir)
    results_csv = results_dir / 'results.csv'
    
    if not results_csv.exists():
        print(f"❌ 未找到results.csv: {results_csv}")
        return
    
    # 读取数据
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # 去除列名空格
    
    print(f"✅ 已加载 {len(df)} 个epoch的数据")
    print(f"📊 列名: {list(df.columns)}")
    
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RT-DETR Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. mAP50 和 mAP50-95
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
    ax1.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP')
    ax1.set_title('Mean Average Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失函数
    ax2 = axes[0, 1]
    loss_cols = [col for col in df.columns if 'loss' in col.lower() and 'train' in col.lower()]
    for col in loss_cols:
        label = col.replace('train/', '').replace('loss', '')
        ax2.plot(df['epoch'], df[col], label=label, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 验证损失
    ax3 = axes[0, 2]
    val_loss_cols = [col for col in df.columns if 'loss' in col.lower() and 'val' in col.lower()]
    if val_loss_cols:
        for col in val_loss_cols:
            label = col.replace('val/', '').replace('loss', '')
            ax3.plot(df['epoch'], df[col], label=label, linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Validation Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No validation loss data', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Precision 和 Recall
    ax4 = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision & Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 学习率
    ax5 = axes[1, 1]
    lr_cols = [col for col in df.columns if 'lr' in col.lower()]
    for col in lr_cols:
        ax5.plot(df['epoch'], df[col], label=col.split('/')[-1], linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 总结信息
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 获取最佳结果
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    best_map50_95 = df.loc[best_epoch, 'metrics/mAP50-95(B)']
    best_map50 = df.loc[best_epoch, 'metrics/mAP50(B)']
    
    summary_text = f"""
    Training Summary
    {'='*30}
    
    Total Epochs: {len(df)}
    
    Best Results (Epoch {best_epoch + 1}):
      • mAP50-95: {best_map50_95:.4f}
      • mAP50: {best_map50:.4f}
    
    Final Results:
      • mAP50-95: {df.iloc[-1]['metrics/mAP50-95(B)']:.4f}
      • mAP50: {df.iloc[-1]['metrics/mAP50(B)']:.4f}
    """
    
    ax6.text(0.1, 0.5, summary_text, 
            transform=ax6.transAxes,
            fontsize=12,
            verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    output_file = results_dir / 'training_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存训练曲线图: {output_file}")
    
    # 也保存独立的mAP图
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2.5, marker='o', markersize=3)
    ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2.5, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('RT-DETR Training - Mean Average Precision', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    map_file = results_dir / 'mAP_curve.png'
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存mAP曲线图: {map_file}")
    
    plt.close('all')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # 默认使用最新的训练目录
        runs_dir = Path('/home/cjj/rtdetr_indoor/runs/detect')
        train_dirs = sorted([d for d in runs_dir.glob('train_*')], key=lambda x: x.stat().st_mtime)
        if train_dirs:
            results_dir = train_dirs[-1]
            print(f"📁 使用最新训练目录: {results_dir}")
        else:
            print("❌ 未找到训练目录")
            sys.exit(1)
    
    generate_training_plots(results_dir)
