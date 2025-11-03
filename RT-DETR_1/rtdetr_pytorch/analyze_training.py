#!/usr/bin/env python
"""
RT-DETR 训练结果分析工具
用法: python analyze_training.py --output_dir /path/to/output
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_training_log(log_file):
    """加载训练日志"""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs

def plot_training_curves(logs, output_dir):
    """绘制训练曲线"""
    epochs = [log['epoch'] for log in logs]
    
    # 提取训练损失
    train_loss = [log.get('train_loss', 0) for log in logs]
    
    # 提取验证指标
    map_vals = []
    map50_vals = []
    map75_vals = []
    
    for log in logs:
        if 'test_coco_eval_bbox' in log:
            bbox_stats = log['test_coco_eval_bbox']
            map_vals.append(bbox_stats[0])
            map50_vals.append(bbox_stats[1])
            map75_vals.append(bbox_stats[2])
        else:
            map_vals.append(0)
            map50_vals.append(0)
            map75_vals.append(0)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练损失
    axes[0, 0].plot(epochs, train_loss, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # mAP
    axes[0, 1].plot(epochs, map_vals, 'g-', linewidth=2, label='mAP')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('Mean Average Precision')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # mAP@50 和 mAP@75
    axes[1, 0].plot(epochs, map50_vals, 'r-', linewidth=2, label='mAP@50')
    axes[1, 0].plot(epochs, map75_vals, 'orange', linewidth=2, label='mAP@75')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('mAP at Different IoU Thresholds')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 所有指标对比
    axes[1, 1].plot(epochs, map_vals, 'g-', linewidth=2, label='mAP')
    axes[1, 1].plot(epochs, map50_vals, 'r--', linewidth=1.5, label='mAP@50')
    axes[1, 1].plot(epochs, map75_vals, 'orange', linewidth=1.5, label='mAP@75')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP')
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {output_dir / 'training_curves.png'}")
    plt.close()

def print_summary(logs):
    """打印训练总结"""
    if not logs:
        print("❌ 没有找到训练日志")
        return
    
    last_log = logs[-1]
    
    print("\n" + "="*80)
    print("🎯 RT-DETR 训练结果总结")
    print("="*80)
    
    print(f"\n📊 训练配置:")
    print(f"  - 总 Epochs: {last_log['epoch'] + 1}")
    print(f"  - 参数量: {last_log['n_parameters']:,}")
    
    if 'test_coco_eval_bbox' in last_log:
        bbox_stats = last_log['test_coco_eval_bbox']
        print(f"\n📈 最终性能 (Epoch {last_log['epoch'] + 1}):")
        print(f"  - mAP      : {bbox_stats[0]:.4f}")
        print(f"  - mAP@50   : {bbox_stats[1]:.4f}")
        print(f"  - mAP@75   : {bbox_stats[2]:.4f}")
        print(f"  - mAP@small: {bbox_stats[3]:.4f}")
        print(f"  - mAP@med  : {bbox_stats[4]:.4f}")
        print(f"  - mAP@large: {bbox_stats[5]:.4f}")
    
    # 找到最佳mAP
    best_map = 0
    best_epoch = 0
    for log in logs:
        if 'test_coco_eval_bbox' in log:
            map_val = log['test_coco_eval_bbox'][0]
            if map_val > best_map:
                best_map = map_val
                best_epoch = log['epoch']
    
    print(f"\n🏆 最佳性能:")
    print(f"  - 最佳 mAP: {best_map:.4f}")
    print(f"  - Epoch: {best_epoch + 1}")
    
    print("\n" + "="*80)
    
    # 训练损失统计
    train_losses = [log.get('train_loss', 0) for log in logs]
    print(f"\n📉 训练损失统计:")
    print(f"  - 初始损失: {train_losses[0]:.4f}")
    print(f"  - 最终损失: {train_losses[-1]:.4f}")
    print(f"  - 平均损失: {np.mean(train_losses):.4f}")
    print(f"  - 最小损失: {np.min(train_losses):.4f}")
    
    print("\n" + "="*80 + "\n")

def analyze_output_dir(output_dir):
    """分析输出目录"""
    output_path = Path(output_dir)
    
    print("\n" + "="*80)
    print(f"📁 分析输出目录: {output_dir}")
    print("="*80 + "\n")
    
    # 检查文件
    files = {
        'log.txt': '训练日志',
        'checkpoint.pth': '最新检查点',
        'eval/latest.pth': 'COCO评估结果',
    }
    
    print("📋 文件清单:")
    for file, desc in files.items():
        file_path = output_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {desc:20s} : {file} ({size:.2f} MB)")
        else:
            print(f"  ❌ {desc:20s} : {file} (不存在)")
    
    # 检查epoch检查点
    checkpoints = sorted(output_path.glob('checkpoint*.pth'))
    if checkpoints:
        print(f"\n  📦 发现 {len(checkpoints)} 个epoch检查点:")
        for ckpt in checkpoints[-5:]:  # 只显示最后5个
            size = ckpt.stat().st_size / (1024 * 1024)
            print(f"     - {ckpt.name} ({size:.2f} MB)")
    
    print("\n" + "="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='RT-DETR训练结果分析')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k',
                        help='训练输出目录')
    parser.add_argument('--plot', action='store_true', help='绘制训练曲线')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"❌ 错误: 输出目录不存在: {output_dir}")
        return
    
    # 分析目录结构
    analyze_output_dir(output_dir)
    
    # 加载日志
    log_file = output_dir / 'log.txt'
    if not log_file.exists():
        print(f"❌ 错误: 未找到训练日志文件: {log_file}")
        return
    
    logs = load_training_log(log_file)
    
    # 打印总结
    print_summary(logs)
    
    # 绘制曲线
    if args.plot:
        try:
            plot_training_curves(logs, output_dir)
        except Exception as e:
            print(f"⚠️  绘图失败: {e}")
            print("提示: 需要安装 matplotlib: pip install matplotlib")

if __name__ == '__main__':
    main()
