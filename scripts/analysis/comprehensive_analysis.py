#!/usr/bin/env python3
"""
RT-DETR训练结果综合分析脚本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_training_results(results_dir):
    """分析训练结果"""
    results_dir = Path(results_dir)
    results_csv = results_dir / 'results.csv'
    
    if not results_csv.exists():
        print(f"❌ 未找到results.csv: {results_csv}")
        return
    
    # 读取数据
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    print("=" * 80)
    print("🎯 RT-DETR训练结果分析报告")
    print("=" * 80)
    print(f"\n📁 训练目录: {results_dir.name}")
    print(f"📊 总训练轮数: {len(df)} epochs")
    print(f"⏱️  总训练时间: {df['time'].iloc[-1] / 3600:.2f} 小时")
    
    # 1. 最佳结果分析
    print("\n" + "="*80)
    print("📈 最佳性能指标")
    print("="*80)
    
    best_map50_95_idx = df['metrics/mAP50-95(B)'].idxmax()
    best_map50_idx = df['metrics/mAP50(B)'].idxmax()
    
    print(f"\n🥇 最佳 mAP50-95:")
    print(f"   Epoch: {best_map50_95_idx + 1}")
    print(f"   mAP50-95: {df.loc[best_map50_95_idx, 'metrics/mAP50-95(B)']:.4f}")
    print(f"   mAP50: {df.loc[best_map50_95_idx, 'metrics/mAP50(B)']:.4f}")
    print(f"   Precision: {df.loc[best_map50_95_idx, 'metrics/precision(B)']:.4f}")
    print(f"   Recall: {df.loc[best_map50_95_idx, 'metrics/recall(B)']:.4f}")
    
    print(f"\n🥇 最佳 mAP50:")
    print(f"   Epoch: {best_map50_idx + 1}")
    print(f"   mAP50: {df.loc[best_map50_idx, 'metrics/mAP50(B)']:.4f}")
    print(f"   mAP50-95: {df.loc[best_map50_idx, 'metrics/mAP50-95(B)']:.4f}")
    
    # 2. 最终结果
    print(f"\n📊 最终结果 (Epoch {len(df)}):")
    final_metrics = {
        'mAP50-95': df.iloc[-1]['metrics/mAP50-95(B)'],
        'mAP50': df.iloc[-1]['metrics/mAP50(B)'],
        'Precision': df.iloc[-1]['metrics/precision(B)'],
        'Recall': df.iloc[-1]['metrics/recall(B)']
    }
    for metric, value in final_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # 3. 训练收敛性分析
    print("\n" + "="*80)
    print("📉 训练收敛性分析")
    print("="*80)
    
    # 损失函数分析
    print("\n🔻 训练损失变化:")
    print(f"   GIoU Loss: {df.iloc[0]['train/giou_loss']:.4f} → {df.iloc[-1]['train/giou_loss']:.4f} (↓{(df.iloc[0]['train/giou_loss'] - df.iloc[-1]['train/giou_loss']):.4f})")
    print(f"   Cls Loss:  {df.iloc[0]['train/cls_loss']:.4f} → {df.iloc[-1]['train/cls_loss']:.4f} (↓{(df.iloc[0]['train/cls_loss'] - df.iloc[-1]['train/cls_loss']):.4f})")
    print(f"   L1 Loss:   {df.iloc[0]['train/l1_loss']:.4f} → {df.iloc[-1]['train/l1_loss']:.4f} (↓{(df.iloc[0]['train/l1_loss'] - df.iloc[-1]['train/l1_loss']):.4f})")
    
    print(f"\n🔻 验证损失变化:")
    print(f"   GIoU Loss: {df.iloc[0]['val/giou_loss']:.4f} → {df.iloc[-1]['val/giou_loss']:.4f}")
    print(f"   Cls Loss:  {df.iloc[0]['val/cls_loss']:.4f} → {df.iloc[-1]['val/cls_loss']:.4f}")
    print(f"   L1 Loss:   {df.iloc[0]['val/l1_loss']:.4f} → {df.iloc[-1]['val/l1_loss']:.4f}")
    
    # 4. 学习率分析
    print("\n" + "="*80)
    print("📚 学习率调度")
    print("="*80)
    print(f"   初始学习率: {df.iloc[0]['lr/pg0']:.6f}")
    print(f"   最终学习率: {df.iloc[-1]['lr/pg0']:.6f}")
    print(f"   衰减比例: {(df.iloc[-1]['lr/pg0'] / df.iloc[0]['lr/pg0'] * 100):.2f}%")
    
    # 5. 训练阶段分析
    print("\n" + "="*80)
    print("📊 训练阶段性能")
    print("="*80)
    
    epochs = len(df)
    stages = {
        '前期 (0-20%)': (0, int(epochs * 0.2)),
        '中期 (20-60%)': (int(epochs * 0.2), int(epochs * 0.6)),
        '后期 (60-100%)': (int(epochs * 0.6), epochs)
    }
    
    for stage_name, (start, end) in stages.items():
        stage_data = df.iloc[start:end]
        print(f"\n{stage_name} (Epoch {start+1}-{end}):")
        print(f"   平均 mAP50-95: {stage_data['metrics/mAP50-95(B)'].mean():.4f}")
        print(f"   最大 mAP50-95: {stage_data['metrics/mAP50-95(B)'].max():.4f}")
        print(f"   平均训练损失: {(stage_data['train/giou_loss'] + stage_data['train/cls_loss'] + stage_data['train/l1_loss']).mean():.4f}")
    
    # 6. 过拟合检测
    print("\n" + "="*80)
    print("🔍 过拟合检测")
    print("="*80)
    
    last_20_epochs = df.tail(20)
    train_loss_trend = (last_20_epochs['train/giou_loss'] + last_20_epochs['train/cls_loss'] + last_20_epochs['train/l1_loss']).values
    val_loss_trend = (last_20_epochs['val/giou_loss'] + last_20_epochs['val/cls_loss'] + last_20_epochs['val/l1_loss']).values
    
    train_loss_slope = np.polyfit(range(len(train_loss_trend)), train_loss_trend, 1)[0]
    val_loss_slope = np.polyfit(range(len(val_loss_trend)), val_loss_trend, 1)[0]
    
    print(f"   训练损失趋势: {'↓ 下降' if train_loss_slope < 0 else '↑ 上升'} ({train_loss_slope:.6f})")
    print(f"   验证损失趋势: {'↓ 下降' if val_loss_slope < 0 else '↑ 上升'} ({val_loss_slope:.6f})")
    
    if val_loss_slope > 0 and abs(val_loss_slope) > abs(train_loss_slope):
        print("   ⚠️  检测到可能的过拟合趋势")
    else:
        print("   ✅ 未检测到明显过拟合")
    
    # 7. 性能稳定性
    print("\n" + "="*80)
    print("📊 性能稳定性分析")
    print("="*80)
    
    map_std = df['metrics/mAP50-95(B)'].tail(20).std()
    map_mean = df['metrics/mAP50-95(B)'].tail(20).mean()
    cv = (map_std / map_mean) * 100 if map_mean > 0 else 0
    
    print(f"   最后20个epoch mAP50-95:")
    print(f"     平均值: {map_mean:.4f}")
    print(f"     标准差: {map_std:.4f}")
    print(f"     变异系数: {cv:.2f}%")
    
    if cv < 2:
        print("   ✅ 性能非常稳定")
    elif cv < 5:
        print("   ✅ 性能较稳定")
    else:
        print("   ⚠️  性能波动较大")
    
    # 8. 训练建议
    print("\n" + "="*80)
    print("💡 训练建议")
    print("="*80)
    
    current_map = df.iloc[-1]['metrics/mAP50-95(B)']
    best_map = df['metrics/mAP50-95(B)'].max()
    
    if current_map < best_map * 0.98:
        print("   ⚠️  当前性能低于最佳性能，建议:")
        print("      - 使用early stopping，避免过度训练")
        print(f"      - 最佳权重在第 {best_map50_95_idx + 1} epoch")
    
    if val_loss_slope > 0:
        print("   ⚠️  验证损失上升，建议:")
        print("      - 增加数据增强强度")
        print("      - 增加weight decay")
        print("      - 考虑使用dropout")
    
    if current_map < 0.3:
        print("   ⚠️  mAP较低，建议:")
        print("      - 检查数据集质量和标注")
        print("      - 增加训练epochs")
        print("      - 尝试更大的模型或预训练权重")
    
    # 9. 性能评级
    print("\n" + "="*80)
    print("⭐ 整体性能评级")
    print("="*80)
    
    score = 0
    factors = []
    
    if best_map >= 0.5:
        score += 30
        factors.append("✅ mAP50-95 >= 0.5 (优秀)")
    elif best_map >= 0.3:
        score += 20
        factors.append("✅ mAP50-95 >= 0.3 (良好)")
    else:
        score += 10
        factors.append("⚠️  mAP50-95 < 0.3 (需改进)")
    
    if cv < 3:
        score += 20
        factors.append("✅ 性能稳定")
    elif cv < 5:
        score += 10
        factors.append("✅ 性能较稳定")
    
    if val_loss_slope < 0:
        score += 20
        factors.append("✅ 持续收敛")
    else:
        score += 5
        factors.append("⚠️  验证损失上升")
    
    if current_map >= best_map * 0.98:
        score += 15
        factors.append("✅ 达到最佳性能")
    else:
        score += 5
    
    if df.iloc[-1]['metrics/precision(B)'] > 0.3 and df.iloc[-1]['metrics/recall(B)'] > 0.25:
        score += 15
        factors.append("✅ Precision & Recall均衡")
    
    print(f"\n   总分: {score}/100")
    print("   评级因素:")
    for factor in factors:
        print(f"   {factor}")
    
    if score >= 80:
        rating = "🏆 优秀 (Excellent)"
    elif score >= 60:
        rating = "✅ 良好 (Good)"
    elif score >= 40:
        rating = "⚠️  及格 (Fair)"
    else:
        rating = "❌ 需改进 (Poor)"
    
    print(f"\n   综合评级: {rating}")
    
    print("\n" + "="*80)
    
    return {
        'best_map50_95': df.loc[best_map50_95_idx, 'metrics/mAP50-95(B)'],
        'best_map50': df.loc[best_map50_idx, 'metrics/mAP50(B)'],
        'final_map50_95': df.iloc[-1]['metrics/mAP50-95(B)'],
        'final_map50': df.iloc[-1]['metrics/mAP50(B)'],
        'score': score,
        'rating': rating
    }

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
    
    results = analyze_training_results(results_dir)
