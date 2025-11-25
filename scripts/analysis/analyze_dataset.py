#!/usr/bin/env python3
"""
数据集质量分析脚本
"""

import os
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def analyze_labels(label_dir):
    """分析标签文件"""
    label_dir = Path(label_dir)
    
    # 统计信息
    total_objects = 0
    class_count = Counter()
    bbox_sizes = []
    objects_per_image = []
    empty_images = 0
    
    label_files = list(label_dir.glob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            empty_images += 1
            objects_per_image.append(0)
            continue
        
        objects_per_image.append(len(lines))
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            total_objects += 1
            class_count[cls_id] += 1
            bbox_sizes.append(width * height)  # 相对面积
    
    return {
        'total_images': len(label_files),
        'total_objects': total_objects,
        'empty_images': empty_images,
        'class_count': dict(class_count),
        'bbox_sizes': bbox_sizes,
        'objects_per_image': objects_per_image,
        'avg_objects': np.mean(objects_per_image) if objects_per_image else 0,
        'avg_bbox_size': np.mean(bbox_sizes) if bbox_sizes else 0,
    }

def main():
    dataset_root = Path('/home/cjj/rtdetr_indoor/datasets/coco_indoor_balanced')
    
    # 读取配置
    yaml_file = dataset_root / 'coco_indoor_balanced.yaml'
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 100)
    print("🔍 COCO Indoor 数据集质量分析")
    print("=" * 100)
    
    # 类别信息
    class_names_dict = config.get('names', {})
    if isinstance(class_names_dict, dict):
        class_names = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
    else:
        class_names = class_names_dict
    
    num_classes = len(class_names)
    
    print(f"\n📋 数据集配置:")
    print(f"   类别数: {num_classes}")
    print(f"   类别名: {', '.join(class_names[:10])}{'...' if len(class_names) > 10 else ''}")
    
    # 分析训练集
    print(f"\n{'='*100}")
    print("📊 训练集分析")
    print("="*100)
    
    train_labels = dataset_root / 'labels' / 'train2017'
    train_stats = analyze_labels(train_labels)
    
    print(f"   总图片数: {train_stats['total_images']}")
    print(f"   总目标数: {train_stats['total_objects']}")
    print(f"   空图片数: {train_stats['empty_images']} ({train_stats['empty_images']/train_stats['total_images']*100:.1f}%)")
    print(f"   平均每图目标数: {train_stats['avg_objects']:.2f}")
    print(f"   平均目标大小: {train_stats['avg_bbox_size']:.4f} (相对面积)")
    
    # 类别分布
    print(f"\n   📈 类别分布 (训练集):")
    sorted_classes = sorted(train_stats['class_count'].items(), key=lambda x: x[1], reverse=True)
    
    for cls_id, count in sorted_classes[:10]:
        if cls_id < len(class_names):
            cls_name = class_names[cls_id]
            percentage = count / train_stats['total_objects'] * 100
            print(f"      • {cls_name:20s}: {count:5d} ({percentage:5.1f}%)")
    
    if len(sorted_classes) > 10:
        print(f"      ... 还有 {len(sorted_classes) - 10} 个类别")
    
    # 分析验证集
    print(f"\n{'='*100}")
    print("📊 验证集分析")
    print("="*100)
    
    val_labels = dataset_root / 'labels' / 'val2017'
    val_stats = analyze_labels(val_labels)
    
    print(f"   总图片数: {val_stats['total_images']}")
    print(f"   总目标数: {val_stats['total_objects']}")
    print(f"   空图片数: {val_stats['empty_images']} ({val_stats['empty_images']/val_stats['total_images']*100:.1f}%)")
    print(f"   平均每图目标数: {val_stats['avg_objects']:.2f}")
    print(f"   平均目标大小: {val_stats['avg_bbox_size']:.4f} (相对面积)")
    
    # 数据集质量评估
    print(f"\n{'='*100}")
    print("⚠️  数据集质量问题诊断")
    print("="*100)
    
    issues = []
    
    # 1. 数据量检查
    if train_stats['total_images'] < 5000:
        issues.append(f"❌ 训练集数量不足: {train_stats['total_images']} < 5000 (建议)")
    
    if val_stats['total_images'] < 1000:
        issues.append(f"❌ 验证集数量不足: {val_stats['total_images']} < 1000 (建议)")
    
    # 2. 类别平衡检查
    if sorted_classes:
        max_count = sorted_classes[0][1]
        min_count = sorted_classes[-1][1]
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 50:
            issues.append(f"❌ 类别严重不平衡: 最多类/{min_count}最少类 = {imbalance_ratio:.1f}x")
        elif imbalance_ratio > 20:
            issues.append(f"⚠️  类别不平衡: 最多类/最少类 = {imbalance_ratio:.1f}x")
    
    # 3. 空图片检查
    empty_ratio_train = train_stats['empty_images'] / train_stats['total_images'] * 100
    if empty_ratio_train > 10:
        issues.append(f"⚠️  训练集空图片过多: {empty_ratio_train:.1f}%")
    
    # 4. 目标密度检查
    if train_stats['avg_objects'] < 2:
        issues.append(f"⚠️  平均目标数过少: {train_stats['avg_objects']:.2f} < 2")
    
    # 5. 目标尺寸检查
    small_objects = sum(1 for size in train_stats['bbox_sizes'] if size < 0.01)
    small_ratio = small_objects / len(train_stats['bbox_sizes']) * 100 if train_stats['bbox_sizes'] else 0
    
    if small_ratio > 50:
        issues.append(f"⚠️  小目标过多: {small_ratio:.1f}% 的目标面积 < 1%")
    
    if issues:
        print("\n   发现以下问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n   ✅ 未发现明显数据集问题")
    
    # 性能影响分析
    print(f"\n{'='*100}")
    print("💡 性能低下原因分析")
    print("="*100)
    
    print("\n   基于数据集分析:")
    
    if train_stats['total_images'] < 5000:
        print("   1. ❌ 训练数据不足")
        print("      • 问题: 仅有 {} 张训练图片,远低于标准数据集规模".format(train_stats['total_images']))
        print("      • 影响: 模型难以学习到充分的特征,容易过拟合")
        print("      • 建议: 收集更多数据或使用强数据增强")
    
    if small_ratio > 50:
        print(f"\n   2. ⚠️  小目标检测困难")
        print(f"      • 问题: {small_ratio:.1f}% 是小目标 (面积<1%)")
        print(f"      • 影响: 小目标检测是计算机视觉难题,严重影响mAP")
        print(f"      • 建议: 使用多尺度训练、增加输入分辨率、FPN等")
    
    if imbalance_ratio > 20:
        print(f"\n   3. ⚠️  类别不平衡")
        print(f"      • 问题: 类别分布不均衡 ({imbalance_ratio:.1f}x)")
        print(f"      • 影响: 模型偏向预测常见类别,忽略罕见类别")
        print(f"      • 建议: 使用Focal Loss、类别权重、过采样等")
    
    if train_stats['avg_objects'] < 3:
        print(f"\n   4. ⚠️  目标稀疏")
        print(f"      • 问题: 平均每图仅 {train_stats['avg_objects']:.2f} 个目标")
        print(f"      • 影响: 训练样本有效性降低")
        print(f"      • 建议: Mosaic增强可提高目标密度")
    
    # 训练配置问题
    print(f"\n   基于训练配置分析:")
    print("   5. ⚠️  训练轮数可能不足")
    print("      • 当前: 89 epochs (在第69轮达到最佳)")
    print("      • 建议: 尝试 120-200 epochs,配合early stopping")
    
    print("\n   6. ⚠️  数据增强可能不足")
    print("      • 建议启用: Mosaic=1.0, MixUp=0.1, Augment=True")
    print("      • 建议增加: degrees=10, translate=0.2, scale=0.9")
    
    print("\n   7. ⚠️  学习率策略")
    print("      • 当前lr=0.0001较保守")
    print("      • 建议: 尝试lr=0.0002-0.0005 + Warmup + Cosine Annealing")
    
    # 改进建议
    print(f"\n{'='*100}")
    print("🎯 具体改进建议")
    print("="*100)
    
    print("\n   优先级 1 - 数据增强 (立即可行):")
    print("      • 启用 Mosaic (mosaic=1.0)")
    print("      • 启用 MixUp (mixup=0.15)")
    print("      • 增加颜色增强 (hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)")
    print("      • 增加几何变换 (degrees=10, scale=0.5)")
    
    print("\n   优先级 2 - 训练策略优化:")
    print("      • 增加训练轮数到 120-150 epochs")
    print("      • 使用 EMA (指数移动平均)")
    print("      • 调整学习率: lr0=0.0002, warmup_epochs=5")
    print("      • 增加 weight_decay=0.0005")
    
    print("\n   优先级 3 - 模型优化:")
    print("      • 尝试更大的输入尺寸 (640 → 800)")
    print("      • 使用预训练权重微调而非从头训练")
    print("      • 考虑使用 RT-DETR-X (更大模型)")
    
    print("\n   优先级 4 - 数据集优化 (长期):")
    print("      • 扩充训练数据到 10000+ 张")
    print("      • 平衡类别分布")
    print("      • 重新标注质量检查")
    print("      • 考虑使用全量COCO数据集")
    
    print("\n" + "="*100)
    
    # 保存统计结果
    output = {
        'train': train_stats,
        'val': val_stats,
        'issues': issues
    }
    
    return output

if __name__ == '__main__':
    results = main()
