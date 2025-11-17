#!/usr/bin/env python3
"""
实验结果分析与可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_results(runs_dir):
    """加载所有训练结果"""
    results = []
    runs_path = Path(runs_dir)
    
    for exp_dir in runs_path.glob('*/'):
        results_csv = exp_dir / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df['experiment'] = exp_dir.name
            results.append(df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return None

def plot_map_comparison(df, save_path):
    """绘制 mAP 对比图"""
    plt.figure(figsize=(12, 6))
    
    # mAP@0.5
    plt.subplot(1, 2, 1)
    for exp in df['experiment'].unique():
        exp_data = df[df['experiment'] == exp]
        plt.plot(exp_data['epoch'], exp_data['metrics/mAP50(B)'], 
                label=exp, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.title('mAP@0.5 Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # mAP@0.5:0.95
    plt.subplot(1, 2, 2)
    for exp in df['experiment'].unique():
        exp_data = df[df['experiment'] == exp]
        plt.plot(exp_data['epoch'], exp_data['metrics/mAP50-95(B)'], 
                label=exp, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5:0.95')
    plt.title('mAP@0.5:0.95 Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ mAP对比图已保存: {save_path}")

def plot_loss_comparison(df, save_path):
    """绘制损失对比图"""
    plt.figure(figsize=(15, 5))
    
    loss_types = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    titles = ['Box Loss', 'Class Loss', 'DFL Loss']
    
    for i, (loss_col, title) in enumerate(zip(loss_types, titles), 1):
        plt.subplot(1, 3, i)
        for exp in df['experiment'].unique():
            exp_data = df[df['experiment'] == exp]
            if loss_col in exp_data.columns:
                plt.plot(exp_data['epoch'], exp_data[loss_col], label=exp)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 损失对比图已保存: {save_path}")

def generate_summary_table(df, save_path):
    """生成汇总表格"""
    summary = []
    
    for exp in df['experiment'].unique():
        exp_data = df[df['experiment'] == exp]
        last_epoch = exp_data.iloc[-1]
        best_map50 = exp_data['metrics/mAP50(B)'].max()
        best_map5095 = exp_data['metrics/mAP50-95(B)'].max()
        
        summary.append({
            'Experiment': exp,
            'Final mAP@0.5': f"{last_epoch['metrics/mAP50(B)']:.4f}",
            'Best mAP@0.5': f"{best_map50:.4f}",
            'Final mAP@0.5:0.95': f"{last_epoch['metrics/mAP50-95(B)']:.4f}",
            'Best mAP@0.5:0.95': f"{best_map5095:.4f}",
            'Final Box Loss': f"{last_epoch['train/box_loss']:.4f}",
            'Final Cls Loss': f"{last_epoch['train/cls_loss']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(save_path, index=False)
    print(f"✅ 汇总表格已保存: {save_path}")
    print("\n📊 实验汇总:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def plot_per_class_ap(runs_dir, save_path):
    """绘制各类别 AP 对比（如果有详细数据）"""
    # 这需要额外的日志记录
    pass

def main():
    """主函数"""
    runs_dir = '/home/cjj/rtdetr_indoor/runs/experiments'
    output_dir = Path(runs_dir) / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("📊 开始分析实验结果...")
    
    # 加载数据
    df = load_results(runs_dir)
    if df is None:
        print("❌ 未找到结果文件")
        return
    
    print(f"✅ 加载了 {len(df['experiment'].unique())} 个实验的数据")
    
    # 生成图表
    plot_map_comparison(df, output_dir / 'map_comparison.png')
    plot_loss_comparison(df, output_dir / 'loss_comparison.png')
    summary_df = generate_summary_table(df, output_dir / 'summary.csv')
    
    # 生成 Markdown 报告
    report_path = output_dir / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write("# RT-DETR 实验分析报告\n\n")
        f.write("## 1. 实验汇总\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## 2. 可视化结果\n\n")
        f.write("### 2.1 mAP 对比\n\n")
        f.write("![mAP Comparison](map_comparison.png)\n\n")
        f.write("### 2.2 损失对比\n\n")
        f.write("![Loss Comparison](loss_comparison.png)\n\n")
        f.write("## 3. 分析结论\n\n")
        f.write("### 最佳模型\n\n")
        best_model = summary_df.loc[summary_df['Best mAP@0.5:0.95'].idxmax()]
        f.write(f"- **模型**: {best_model['Experiment']}\n")
        f.write(f"- **mAP@0.5:0.95**: {best_model['Best mAP@0.5:0.95']}\n")
        f.write("\n### 改进建议\n\n")
        f.write("基于实验结果，建议...\n")
    
    print(f"\n✅ 分析完成！报告保存在: {report_path}")

if __name__ == "__main__":
    main()
