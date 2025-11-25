#!/usr/bin/env python3
"""
数据集优化脚本：解决小目标过多的问题
策略：
1. 删除极小目标 (面积<0.003)
2. 对很小目标进行扩大处理 (0.003-0.01 → ×1.5倍)
3. 删除只有小目标的图片
4. 保留多目标图片以提高学习效率
"""

import shutil
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import random
from tqdm import tqdm

def analyze_and_optimize_dataset(source_dir, output_dir, 
                                 min_area_threshold=0.003,
                                 small_area_threshold=0.01,
                                 min_obj_per_image=1):
    """
    优化数据集
    
    Args:
        source_dir: 原始数据集
        output_dir: 输出目录
        min_area_threshold: 删除小于此值的bbox
        small_area_threshold: 扩大此值以下的bbox
        min_obj_per_image: 图片最少保留的目标数
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    label_dir = source_path / "labels" / "train2017"
    img_dir = source_path / "images" / "train2017"
    
    print("="*80)
    print("📊 数据集优化 - 解决小目标过多问题")
    print("="*80)
    
    # 阶段1: 分析所有图片
    print(f"\n📋 分析所有图片...")
    
    image_stats = {}  # image_id -> {bboxes, areas, valid_bboxes}
    total_original_bboxes = 0
    total_small_bboxes = 0
    total_tiny_bboxes = 0
    
    for label_file in tqdm(label_dir.glob("*.txt"), desc="Analyzing"):
        image_id = label_file.stem
        bboxes = []
        areas = []
        
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    area = w * h
                    
                    bboxes.append((cls_id, x, y, w, h, area))
                    areas.append(area)
                    total_original_bboxes += 1
                    
                    if area < min_area_threshold:
                        total_tiny_bboxes += 1
                    elif area < small_area_threshold:
                        total_small_bboxes += 1
        
        image_stats[image_id] = {
            "bboxes": bboxes,
            "areas": np.array(areas),
            "original_count": len(bboxes)
        }
    
    print(f"\n✅ 分析完成")
    print(f"   总bbox数: {total_original_bboxes}")
    print(f"   极小bbox (<{min_area_threshold}): {total_tiny_bboxes} ({total_tiny_bboxes/total_original_bboxes*100:.1f}%)")
    print(f"   小bbox ({min_area_threshold}-{small_area_threshold}): {total_small_bboxes} ({total_small_bboxes/total_original_bboxes*100:.1f}%)")
    
    # 阶段2: 优化处理
    print(f"\n🔧 优化处理...")
    
    kept_images = 0
    removed_images = 0
    total_kept_bboxes = 0
    total_removed_bboxes = 0
    total_expanded_bboxes = 0
    
    kept_image_ids = []
    
    for image_id, stats in tqdm(image_stats.items(), desc="Optimizing"):
        bboxes = stats["bboxes"]
        new_bboxes = []
        
        for cls_id, x, y, w, h, area in bboxes:
            # 步骤1: 删除极小目标
            if area < min_area_threshold:
                total_removed_bboxes += 1
                continue
            
            # 步骤2: 扩大小目标
            if area < small_area_threshold:
                # 扩大1.5倍
                scale = np.sqrt(1.5)
                w = min(1.0, w * scale)
                h = min(1.0, h * scale)
                x = max(0, min(1.0, x))
                y = max(0, min(1.0, y))
                total_expanded_bboxes += 1
            
            new_bboxes.append((cls_id, x, y, w, h))
            total_kept_bboxes += 1
        
        # 步骤3: 只保留有足够目标的图片
        if len(new_bboxes) >= min_obj_per_image:
            kept_images += 1
            kept_image_ids.append(image_id)
            # 暂存新的bbox信息
            image_stats[image_id]["optimized_bboxes"] = new_bboxes
        else:
            removed_images += 1
    
    print(f"\n✅ 优化完成")
    print(f"   保留图片: {kept_images} / {len(image_stats)} (-{removed_images})")
    print(f"   保留bbox: {total_kept_bboxes} / {total_original_bboxes} (-{total_removed_bboxes})")
    print(f"   扩大bbox: {total_expanded_bboxes}")
    
    # 阶段3: 复制文件
    print(f"\n📁 复制优化后的数据...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_img_dir = output_path / "images" / "train2017"
    output_label_dir = output_path / "labels" / "train2017"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    for image_id in tqdm(kept_image_ids, desc="Copy files"):
        # 复制图片
        src_img = img_dir / f"{image_id}.jpg"
        dst_img = output_img_dir / f"{image_id}.jpg"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        
        # 写入优化后的标注
        dst_label = output_label_dir / f"{image_id}.txt"
        with open(dst_label, 'w') as f:
            for cls_id, x, y, w, h in image_stats[image_id]["optimized_bboxes"]:
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    # 阶段4: 验证
    print(f"\n✅ 验证优化结果...")
    
    bbox_areas_new = []
    total_images_new = 0
    
    for label_file in output_label_dir.glob("*.txt"):
        total_images_new += 1
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    bbox_areas_new.append(w * h)
    
    bbox_areas_new = np.array(bbox_areas_new)
    
    print(f"\n   新数据集统计:")
    print(f"   总图片: {total_images_new}")
    print(f"   总bbox: {len(bbox_areas_new)}")
    print(f"   平均bbox/图: {len(bbox_areas_new)/total_images_new:.2f}")
    
    print(f"\n   小目标分布 (优化后):")
    print(f"   极小目标 (<0.005): {sum(bbox_areas_new < 0.005)/len(bbox_areas_new)*100:.1f}%")
    print(f"   很小目标 (0.005-0.01): {sum((bbox_areas_new >= 0.005) & (bbox_areas_new < 0.01))/len(bbox_areas_new)*100:.1f}%")
    print(f"   小目标 (0.01-0.05): {sum((bbox_areas_new >= 0.01) & (bbox_areas_new < 0.05))/len(bbox_areas_new)*100:.1f}%")
    print(f"   中等目标 (0.05-0.2): {sum((bbox_areas_new >= 0.05) & (bbox_areas_new < 0.2))/len(bbox_areas_new)*100:.1f}%")
    print(f"   大目标 (>0.2): {sum(bbox_areas_new >= 0.2)/len(bbox_areas_new)*100:.1f}%")
    
    original_small = sum((np.array(list(np.concatenate([stats["areas"] for stats in image_stats.values()]))) < 0.01)) / \
                     sum(len(stats["areas"]) for stats in image_stats.values()) * 100
    new_small = sum((bbox_areas_new < 0.01)) / len(bbox_areas_new) * 100
    improvement = (original_small - new_small) / original_small * 100
    
    print(f"\n🎯 改进效果:")
    print(f"   原始小目标占比: {original_small:.1f}%")
    print(f"   优化后小目标占比: {new_small:.1f}%")
    print(f"   改进幅度: {improvement:.1f}% ✅")
    
    # 复制配置文件
    yaml_src = source_path / "coco_indoor_balanced_balanced.yaml"
    yaml_dst = output_path / "coco_indoor_balanced_optimized.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, yaml_dst)
    
    print(f"\n🎉 完成！优化后的数据集保存到: {output_path}")
    print(f"   配置: {yaml_dst}")
    print(f"\n💡 下一步: 用新数据集训练")
    print(f"   python3 scripts/training/auto_train_rtdetr.py \\")
    print(f"     --dataset coco_indoor_balanced_optimized \\")
    print(f"     --skip-confirm")

if __name__ == "__main__":
    import sys
    
    source = sys.argv[1] if len(sys.argv) > 1 else "datasets/coco_indoor_balanced_balanced"
    output = sys.argv[2] if len(sys.argv) > 2 else "datasets/coco_indoor_balanced_optimized"
    
    print(f"源数据集: {source}")
    print(f"输出数据集: {output}\n")
    
    random.seed(42)
    np.random.seed(42)
    
    analyze_and_optimize_dataset(
        source,
        output,
        min_area_threshold=0.003,      # 删除<0.3%面积的bbox
        small_area_threshold=0.01,     # 扩大<1%面积的bbox
        min_obj_per_image=1            # 每张图至少1个目标
    )
