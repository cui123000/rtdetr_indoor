#!/usr/bin/env python3
"""
严格平衡数据集脚本 - 精确减少到10000张图片以内，保持类别平衡
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import yaml

def load_class_names():
    """从data.yaml读取类别名称"""
    with open("datasets/coco_indoor_balanced/data.yaml", 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def analyze_dataset():
    """分析数据集的类别分布"""
    class_to_images = defaultdict(set)  # {class_id: {image_file}}
    image_to_classes = {}  # {image_file: [class_ids]}
    
    labels_dir = "datasets/coco_indoor_balanced/labels"
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            image_name = label_file.replace('.txt', '')
            label_path = os.path.join(labels_dir, label_file)
            
            classes_in_image = set()
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_to_images[class_id].add(image_name)
                        classes_in_image.add(class_id)
            
            image_to_classes[image_name] = classes_in_image
    
    return class_to_images, image_to_classes

def select_balanced_images_strict(class_to_images, image_to_classes, 
                                  target_images=9000, target_classes=20):
    """
    严格选择平衡的图片集合
    - 目标图片数: 9000
    - 目标类别数: 20 (每个类别平均450张图片)
    """
    
    class_names = load_class_names()
    
    # 1. 按出现频率排序类别
    class_sizes = {cid: len(imgs) for cid, imgs in class_to_images.items()}
    sorted_classes = sorted(class_sizes.items(), key=lambda x: -x[1])
    
    print(f"原始数据集:")
    print(f"  总类别数: {len(class_to_images)}")
    print(f"  总图片数: {len(image_to_classes)}")
    print(f"  类别分布范围: {min(class_sizes.values())} - {max(class_sizes.values())} 张图片")
    
    # 2. 选择最常见的N个类别
    selected_classes = [cid for cid, _ in sorted_classes[:target_classes]]
    selected_classes.sort()
    
    print(f"\n目标配置:")
    print(f"  目标图片数: {target_images}")
    print(f"  目标类别数: {target_classes}")
    print(f"  每个类别平均: {target_images // target_classes} 张图片")
    
    # 3. 逐类别严格采样
    imgs_per_class = target_images // target_classes
    selected_images = set()
    class_sampling_info = {}
    
    print(f"\n类别采样计划:")
    for class_id in selected_classes:
        class_name = class_names.get(class_id, f"class_{class_id}")
        available_imgs = class_to_images[class_id]
        
        # 严格采样
        sample_size = min(imgs_per_class, len(available_imgs))
        sampled = set(random.sample(list(available_imgs), sample_size))
        selected_images.update(sampled)
        
        class_sampling_info[class_id] = {
            'name': class_name,
            'sampled': len(sampled),
            'available': len(available_imgs),
            'target': imgs_per_class
        }
        
        print(f"  {class_name:20} (id={class_id:2}): {len(sampled):4} / {imgs_per_class} "
              f"(可用: {len(available_imgs):6})")
    
    final_count = len(selected_images)
    print(f"\n最终选择:")
    print(f"  总图片数: {final_count}")
    print(f"  目标类别: {len(selected_classes)}")
    
    # 确保不超过目标
    if final_count > target_images:
        print(f"\n⚠️ 超过目标 ({final_count} > {target_images})，进行微调...")
        # 随机移除超出部分
        excess = final_count - target_images
        to_remove = random.sample(list(selected_images), excess)
        for img in to_remove:
            selected_images.remove(img)
        final_count = len(selected_images)
        print(f"   移除 {excess} 张图片，最终: {final_count}")
    
    return selected_images, selected_classes, class_sampling_info

def create_balanced_dataset(selected_images, selected_classes):
    """创建平衡的数据集"""
    
    output_dir = "datasets/coco_indoor_strict"
    images_dir = os.path.join(output_dir, "images")
    labels_dir_out = os.path.join(output_dir, "labels")
    
    # 创建目录
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir_out, exist_ok=True)
    
    # 复制图片和标签
    src_images_dir = "datasets/coco_indoor_balanced/images"
    src_labels_dir = "datasets/coco_indoor_balanced/labels"
    
    print(f"\n复制文件到 {output_dir}...")
    
    for i, image_name in enumerate(sorted(selected_images)):
        if (i + 1) % 1000 == 0:
            print(f"  进度: {i + 1} / {len(selected_images)}")
        
        # 找到原始图片文件
        src_img = None
        if os.path.exists(os.path.join(src_images_dir, f"{image_name}.jpg")):
            src_img = os.path.join(src_images_dir, f"{image_name}.jpg")
        else:
            # 可能在子目录中
            for root, dirs, files in os.walk(src_images_dir):
                if f"{image_name}.jpg" in files:
                    src_img = os.path.join(root, f"{image_name}.jpg")
                    break
        
        if src_img and os.path.exists(src_img):
            dst_img = os.path.join(images_dir, f"{image_name}.jpg")
            shutil.copy2(src_img, dst_img)
        
        # 复制和过滤标签（只保留选中的类别）
        src_label = os.path.join(src_labels_dir, f"{image_name}.txt")
        dst_label = os.path.join(labels_dir_out, f"{image_name}.txt")
        
        if os.path.exists(src_label):
            with open(src_label, 'r') as f:
                lines = f.readlines()
            
            # 过滤类别
            filtered_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id in selected_classes:
                        filtered_lines.append(line)
            
            if filtered_lines:
                with open(dst_label, 'w') as f:
                    f.writelines(filtered_lines)
    
    print(f"  完成！复制了 {len(selected_images)} 张图片")
    
    return output_dir, selected_classes

def create_dataset_yaml(output_dir, selected_classes):
    """创建data.yaml配置文件"""
    
    class_names = load_class_names()
    
    # 创建新的类别映射（0-19）
    new_names = {}
    class_mapping = {}  # 旧ID -> 新ID
    
    for new_id, old_id in enumerate(sorted(selected_classes)):
        new_names[new_id] = class_names[old_id]
        class_mapping[old_id] = new_id
    
    # 更新标签文件中的类别ID
    labels_dir = os.path.join(output_dir, "labels")
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    old_class_id = int(parts[0])
                    new_class_id = class_mapping[old_class_id]
                    parts[0] = str(new_class_id)
                    updated_lines.append(' '.join(parts) + '\n')
            
            with open(label_path, 'w') as f:
                f.writelines(updated_lines)
    
    # 创建data.yaml
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images',
        'val': 'images',
        'nc': len(selected_classes),
        'names': new_names
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n生成配置文件: {yaml_path}")
    print(f"  类别数: {len(selected_classes)}")
    
    return new_names, class_mapping

def create_metadata(output_dir, selected_images, selected_classes, new_names, class_sampling_info):
    """创建元数据文件"""
    
    metadata = {
        "method": "Strict balanced sampling from auto-classified indoor COCO",
        "source_dataset": "coco_indoor_balanced",
        "target_images": 9000,
        "target_classes": len(selected_classes),
        "actual_images": len(selected_images),
        "selected_classes": sorted(selected_classes),
        "class_names": new_names,
        "class_sampling_info": class_sampling_info,
        "notes": "严格控制数据量在9000张以内，每个类别均衡采样"
    }
    
    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"生成元数据文件: {meta_path}")

def main():
    random.seed(42)
    
    # 分析现有数据集
    print("=" * 60)
    print("严格平衡数据集生成工具")
    print("=" * 60)
    
    class_to_images, image_to_classes = analyze_dataset()
    
    # 选择平衡的图片 - 目标9000张，20个类别
    selected_images, selected_classes, class_sampling_info = select_balanced_images_strict(
        class_to_images, 
        image_to_classes,
        target_images=9000,
        target_classes=20
    )
    
    # 创建平衡数据集
    output_dir, final_classes = create_balanced_dataset(selected_images, selected_classes)
    
    # 创建yaml配置
    new_names, class_mapping = create_dataset_yaml(output_dir, final_classes)
    
    # 创建元数据
    create_metadata(output_dir, selected_images, final_classes, new_names, class_sampling_info)
    
    # 最终统计
    print(f"\n" + "=" * 60)
    print(f"完成！严格平衡数据集已生成")
    print(f"=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"图片数: {len(selected_images)}")
    print(f"类别数: {len(final_classes)}")
    print(f"\n选中的类别:")
    for new_id, class_name in new_names.items():
        print(f"  {new_id:2}: {class_name}")

if __name__ == '__main__':
    main()
