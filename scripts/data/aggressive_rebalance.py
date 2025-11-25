#!/usr/bin/env python3
"""
激进的数据集平衡策略：
- Person: 只保留600张（从3868→600）
- 其他类：保留所有，每类不超过均值×1.5倍
- 总数目标：3500-4500张，完全平衡
"""

import shutil
import json
from pathlib import Path
from collections import Counter, defaultdict
import random
import argparse
from tqdm import tqdm

def analyze_dataset(label_dir):
    """分析数据集"""
    label_counter = Counter()
    image_to_bboxes = {}
    
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        bboxes = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                label_counter[cls_id] += 1
                bboxes.append(cls_id)
        
        image_to_bboxes[label_file.stem] = bboxes
    
    return label_counter, image_to_bboxes

def get_class_distribution(label_dir):
    """获取每个类别的图片列表"""
    class_to_images = defaultdict(list)
    
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                class_to_images[cls_id].append(label_file.stem)
    
    for cls_id in class_to_images:
        class_to_images[cls_id] = list(set(class_to_images[cls_id]))
    
    return class_to_images

def aggressive_rebalance(source_dir, output_dir, target_images=4000):
    """
    激进平衡：
    1. Person类只保留最少数量
    2. 其他类别均匀采样
    3. 总数控制在target_images
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📊 分析原始数据集...")
    label_dir = source_path / "labels" / "train2017"
    img_dir = source_path / "images" / "train2017"
    
    label_counter, image_to_bboxes = analyze_dataset(label_dir)
    class_to_images = get_class_distribution(label_dir)
    
    total_bboxes = sum(label_counter.values())
    
    print(f"\n原始统计:")
    print(f"  总图片数: {len(image_to_bboxes)}")
    print(f"  总bbox数: {total_bboxes}")
    print(f"  Person类: {label_counter[19]} bbox ({label_counter[19]/total_bboxes*100:.1f}%)")
    
    # 激进策略：取每类图片均匀采样
    print(f"\n🎯 激进平衡策略:")
    
    # Person: 只要600张
    person_images = class_to_images[19]
    num_person_to_keep = min(600, len(person_images))
    selected_person = random.sample(person_images, num_person_to_keep)
    print(f"  Person: {len(person_images)} → {num_person_to_keep} 图片")
    
    # 其他类：均匀采样
    selected_images = set(selected_person)
    
    other_classes = [c for c in label_counter.keys() if c != 19]
    num_other_classes = len(other_classes)
    target_per_class = (target_images - num_person_to_keep) // num_other_classes
    
    print(f"  其他{num_other_classes}类: 每类最多{target_per_class}张图片")
    
    for cls_id in other_classes:
        class_images = class_to_images[cls_id]
        num_to_keep = min(target_per_class, len(class_images))
        selected = random.sample(class_images, num_to_keep)
        selected_images.update(selected)
        print(f"    Class {cls_id:2d}: {len(class_images):4d} → {num_to_keep:4d} 图片")
    
    print(f"\n  总选择: {len(selected_images)} 图片")
    
    # 创建输出目录
    output_img_dir = output_path / "images" / "train2017"
    output_label_dir = output_path / "labels" / "train2017"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    print(f"\n📋 复制文件...")
    for img_name in tqdm(selected_images, desc="Copy"):
        src_img = img_dir / f"{img_name}.jpg"
        dst_img = output_img_dir / f"{img_name}.jpg"
        
        src_label = label_dir / f"{img_name}.txt"
        dst_label = output_label_dir / f"{img_name}.txt"
        
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
    
    # 验证
    print(f"\n✅ 验证新数据集...")
    new_label_counter, new_image_to_bboxes = analyze_dataset(output_label_dir)
    new_total_bboxes = sum(new_label_counter.values())
    
    print(f"\n新数据集统计:")
    print(f"  总图片数: {len(new_image_to_bboxes)}")
    print(f"  总bbox数: {new_total_bboxes}")
    print(f"\n  类别分布:")
    print(f"  {'Class':>6} | {'Bbox Count':>10} | {'Percentage':>10} | {'Images':>10}")
    print(f"  {'-'*50}")
    
    for cls_id in sorted(new_label_counter.keys()):
        count = new_label_counter[cls_id]
        ratio = count / new_total_bboxes * 100
        img_count = len([im for im in new_image_to_bboxes.values() if cls_id in im])
        marker = " ← PERSON" if cls_id == 19 else ""
        print(f"  {cls_id:6d} | {count:10d} | {ratio:9.1f}% | {img_count:10d}{marker}")
    
    min_count = min(new_label_counter.values())
    max_count = max(new_label_counter.values())
    imbalance = max_count / min_count if min_count > 0 else 0
    
    new_person_ratio = new_label_counter[19] / new_total_bboxes * 100
    print(f"\n📊 平衡效果:")
    print(f"  不平衡比: {imbalance:.2f}x (原: 24.09x)")
    print(f"  Person占比: {new_person_ratio:.1f}% (原: 33.6%)")
    print(f"  平衡提升: {(1 - imbalance/24.09)*100:.1f}% ✅")
    
    # 复制配置文件
    yaml_src = source_path / "coco_indoor_balanced.yaml"
    yaml_dst = output_path / "coco_indoor_balanced_balanced.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, yaml_dst)
    
    print(f"\n🎉 数据集重新平衡完成！")
    print(f"   输出: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/coco_indoor_balanced")
    parser.add_argument("--output", default="datasets/coco_indoor_balanced_balanced")
    parser.add_argument("--target-images", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    aggressive_rebalance(args.source, args.output, target_images=args.target_images)
