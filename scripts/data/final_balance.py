#!/usr/bin/env python3
"""
最终方案：以无person图片为主骨架 + 少量person图片
- 无person图片: 3000张（主骨架，保留所有其他类别）
- 单/少person: 1000张（保留person检测能力）
- 总数: 4000张
- 最终person占比: ~15%
"""

import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random
from tqdm import tqdm

def build_final_balanced_dataset(source_dir, output_dir):
    """最终平衡方案"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    label_dir = source_path / "labels" / "train2017"
    img_dir = source_path / "images" / "train2017"
    
    print("📊 分类图片...")
    
    # 分类
    no_person_imgs = []
    person_imgs = []  # 1-2个person
    
    for label_file in sorted(label_dir.glob("*.txt")):
        image_id = label_file.stem
        person_count = 0
        
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 19:
                    person_count += 1
        
        if person_count == 0:
            no_person_imgs.append(image_id)
        elif person_count <= 2:
            person_imgs.append(image_id)
    
    print(f"  无person图: {len(no_person_imgs)}")
    print(f"  1-2人图: {len(person_imgs)}")
    
    # 采样
    print(f"\n🎯 采样策略...")
    
    # 主骨架：所有无person图（最多3500张）
    no_person_sampled = random.sample(no_person_imgs, min(3500, len(no_person_imgs)))
    print(f"  无person: {len(no_person_sampled)} 张")
    
    # 补充：1-2人图
    person_needed = 4000 - len(no_person_sampled)
    person_sampled = random.sample(person_imgs, min(person_needed, len(person_imgs)))
    print(f"  1-2人: {len(person_sampled)} 张")
    
    selected = set(no_person_sampled) | set(person_sampled)
    print(f"  总数: {len(selected)} 张")
    
    # 创建输出
    output_path.mkdir(parents=True, exist_ok=True)
    output_img_dir = output_path / "images" / "train2017"
    output_label_dir = output_path / "labels" / "train2017"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 复制文件...")
    for img_id in tqdm(selected):
        src_img = img_dir / f"{img_id}.jpg"
        dst_img = output_img_dir / f"{img_id}.jpg"
        src_label = label_dir / f"{img_id}.txt"
        dst_label = output_label_dir / f"{img_id}.txt"
        
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
    
    # 验证
    print(f"\n✅ 验证...")
    
    class_counter = Counter()
    img_count = 0
    
    for label_file in output_label_dir.glob("*.txt"):
        img_count += 1
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    class_counter[cls_id] += 1
    
    total_bbox = sum(class_counter.values())
    
    print(f"\n  总图片: {img_count}")
    print(f"  总bbox: {total_bbox}")
    print(f"\n  类别分布:")
    print(f"  {'Class':>6} | {'Bbox':>6} | {'%':>6}")
    print(f"  {'-'*24}")
    
    for cls_id in sorted(class_counter.keys()):
        count = class_counter[cls_id]
        ratio = count / total_bbox * 100
        marker = " ← PERSON" if cls_id == 19 else ""
        print(f"  {cls_id:6d} | {count:6d} | {ratio:5.1f}%{marker}")
    
    min_c = min(class_counter.values())
    max_c = max(class_counter.values())
    imbalance = max_c / min_c
    person_ratio = class_counter.get(19, 0) / total_bbox * 100
    
    print(f"\n📊 平衡对比:")
    print(f"  原始: 24.09x不平衡 | Person 33.6%")
    print(f"  现在: {imbalance:.2f}x不平衡 | Person {person_ratio:.1f}%")
    print(f"  改进: {(1-imbalance/24.09)*100:.0f}% ✅")
    
    # 复制yaml
    yaml_src = source_path / "coco_indoor_balanced.yaml"
    yaml_dst = output_path / "coco_indoor_balanced_balanced.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, yaml_dst)
    
    print(f"\n🎉 完成！")
    print(f"   输出: {output_path}")

if __name__ == "__main__":
    random.seed(42)
    build_final_balanced_dataset(
        "datasets/coco_indoor_balanced",
        "datasets/coco_indoor_balanced_balanced"
    )
