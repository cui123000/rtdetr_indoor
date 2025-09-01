#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从LVIS或COCO数据集创建室内场景推理数据集
- 支持多种图像ID映射方式
- 自动查找匹配的图像文件
- 更简单的场景分类
"""

import os
import json
import argparse
import random
import shutil
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

# 定义室内物体类别ID (基于COCO/LVIS类别)
INDOOR_OBJECTS = {
    62: "chair",
    63: "couch/sofa",
    64: "potted plant",
    65: "bed",
    66: "dining table",
    67: "toilet",
    68: "tv",
    69: "laptop",
    70: "mouse",
    71: "remote",
    72: "keyboard",
    73: "cell phone",
    74: "microwave",
    75: "oven",
    76: "toaster",
    77: "sink",
    78: "refrigerator",
    79: "book",
    80: "clock",
    81: "vase",
    83: "teddy bear",
    84: "hair drier",
    85: "toothbrush"
}

def find_image_file(image_id, image_dir, original_filename=None):
    """
    尝试多种方式查找与图像ID匹配的文件
    
    Args:
        image_id: 图像ID
        image_dir: 图像目录
        original_filename: 原始文件名
        
    Returns:
        找到的图像路径或None
    """
    # 检查原始文件名
    if original_filename and os.path.exists(os.path.join(image_dir, original_filename)):
        return os.path.join(image_dir, original_filename)
    
    # 尝试多种常见文件名格式
    patterns = [
        f"{image_id:012d}.jpg",           # 12位补零: 000000123456.jpg
        f"{image_id:012d}.jpeg",          # 12位补零jpeg
        f"{image_id:08d}.jpg",            # 8位补零: 00123456.jpg
        f"{image_id}.jpg",                # 无补零: 123456.jpg
        f"COCO_val2017_{image_id:012d}.jpg", # COCO训练集格式
        f"COCO_train2017_{image_id:012d}.jpg", # COCO训练集格式
        f"COCO_test2017_{image_id:012d}.jpg",  # COCO测试集格式
        f"{image_id:012d}.*",             # 任何扩展名
        f"{image_id}.*"                   # 无补零，任何扩展名
    ]
    
    for pattern in patterns:
        if "*" in pattern:
            # 使用glob查找匹配文件
            matches = glob.glob(os.path.join(image_dir, pattern))
            if matches:
                return matches[0]
        else:
            # 直接检查文件是否存在
            file_path = os.path.join(image_dir, pattern)
            if os.path.exists(file_path):
                return file_path
    
    # 如果图像目录有子目录，搜索子目录
    subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    for subdir in subdirs:
        for pattern in patterns:
            if "*" in pattern:
                matches = glob.glob(os.path.join(image_dir, subdir, pattern))
                if matches:
                    return matches[0]
            else:
                file_path = os.path.join(image_dir, subdir, pattern)
                if os.path.exists(file_path):
                    return file_path
    
    return None

# 修改 extract_indoor_images 函数，增加更严格的筛选条件

def extract_indoor_images(dataset, annotations_by_image_id, min_indoor_objects=2, min_indoor_types=2, min_confidence=0.7):
    """
    从数据集中提取包含室内物体的图像，使用更严格的筛选条件
    
    Args:
        dataset: 数据集字典
        annotations_by_image_id: 按图像ID索引的标注
        min_indoor_objects: 最少需要包含的室内物体数量
        min_indoor_types: 最少需要包含的不同类型室内物体数量
        min_confidence: 最低置信度要求(如果标注中包含分数)
        
    Returns:
        室内图像列表
    """
    indoor_images = []
    
    # 定义核心室内物体 - 这些物体的存在更能确定是室内场景
    CORE_INDOOR_OBJECTS = {
        63: "couch/sofa",     # 沙发最能表示客厅
        65: "bed",            # 床最能表示卧室
        66: "dining table",   # 餐桌最能表示餐厅
        67: "toilet",         # 马桶最能表示卫生间
        68: "tv",             # 电视通常在室内
        74: "microwave",      # 微波炉表示厨房
        75: "oven",           # 烤箱表示厨房
        78: "refrigerator"    # 冰箱表示厨房
    }
    
    # 定义室外可能也出现的物体 - 这些物体存在不一定表示室内
    AMBIGUOUS_OBJECTS = {
        62: "chair",          # 椅子室内室外都有
        64: "potted plant",   # 盆栽植物室内室外都有
        73: "cell phone",     # 手机不能确定环境
        79: "book",           # 书不能确定环境
        81: "vase"            # 花瓶不能确定环境
    }
    
    for img in tqdm(dataset['images'], desc="严格筛选室内图像"):
        img_id = img['id']
        if img_id not in annotations_by_image_id:
            continue
            
        annotations = annotations_by_image_id[img_id]
        
        # 过滤掉低置信度的标注(如果有置信度字段)
        valid_annotations = []
        for ann in annotations:
            # 检查是否有置信度字段
            if 'score' in ann and ann['score'] < min_confidence:
                continue
            valid_annotations.append(ann)
        
        if not valid_annotations:
            continue
            
        # 计算包含的室内物体
        indoor_object_count = 0
        indoor_object_types = set()
        core_indoor_count = 0
        ambiguous_count = 0
        
        for ann in valid_annotations:
            cat_id = ann['category_id']
            
            # 统计核心室内物体
            if cat_id in CORE_INDOOR_OBJECTS:
                indoor_object_count += 1
                indoor_object_types.add(cat_id)
                core_indoor_count += 1
                
            # 统计一般室内物体
            elif cat_id in INDOOR_OBJECTS:
                indoor_object_count += 1
                indoor_object_types.add(cat_id)
                
            # 统计模糊物体(室内室外都可能出现的)
            if cat_id in AMBIGUOUS_OBJECTS:
                ambiguous_count += 1
        
        # 严格筛选标准:
        # 1. 至少有指定数量的室内物体
        # 2. 至少有指定数量的不同类型室内物体
        # 3. 至少有一个核心室内物体，或室内物体数量显著多于模糊物体
        is_indoor = (
            indoor_object_count >= min_indoor_objects and 
            len(indoor_object_types) >= min_indoor_types and
            (core_indoor_count > 0 or indoor_object_count >= ambiguous_count * 2)
        )
        
        if is_indoor:
            # 计算室内场景置信度分数
            confidence_score = (
                0.3 * core_indoor_count +                # 核心室内物体贡献较大权重
                0.2 * (len(indoor_object_types) - core_indoor_count) +  # 其他室内物体类型数量
                0.1 * (indoor_object_count - len(indoor_object_types))  # 额外的室内物体实例
            )
            
            if core_indoor_count >= 2:  # 有多个核心室内物体时额外加分
                confidence_score += 0.3
                
            # 归一化得分到0-1范围
            confidence_score = min(1.0, confidence_score)
            
            indoor_images.append({
                'image': img,
                'annotations': valid_annotations,
                'indoor_object_count': indoor_object_count,
                'indoor_object_types': indoor_object_types,
                'core_indoor_count': core_indoor_count,
                'confidence_score': confidence_score
            })
    
    # 按室内场景置信度排序
    indoor_images.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    print(f"找到 {len(indoor_images)} 张高置信度室内场景图像")
    return indoor_images
def create_inference_dataset(args):
    """
    创建室内场景推理数据集
    
    Args:
        args: 命令行参数
    """
    # 加载数据集
    print(f"加载数据集: {args.input}")
    with open(args.input, 'r') as f:
        dataset = json.load(f)
    
    # 创建输出目录
    output_dir = os.path.join(args.output, "indoor_training")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图像目录
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 按图像ID组织标注
    print("索引数据集标注...")
    annotations_by_image_id = {}
    for ann in dataset['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)
    
    # 提取室内图像
    indoor_images = extract_indoor_images(
        dataset, 
        annotations_by_image_id, 
        min_indoor_objects=args.min_objects,
        min_indoor_types=args.min_types,
        min_confidence=args.min_confidence
    )
    
    # 选择高置信度的图像
    if args.threshold > 0:
        filtered_images = [img for img in indoor_images if img['confidence_score'] >= args.threshold]
        print(f"按置信度阈值({args.threshold})筛选后，剩余 {len(filtered_images)}/{len(indoor_images)} 张图像")
        indoor_images = filtered_images
    
    
    # 打乱顺序并限制数量
    if args.max_images > 0 and len(indoor_images) > args.max_images:
        print(f"随机选择 {args.max_images} 张图像")
        random.shuffle(indoor_images)
        indoor_images = indoor_images[:args.max_images]
    
    # 创建新的数据集
    new_dataset = {
        'info': dataset.get('info', {}),
        'licenses': dataset.get('licenses', []),
        'categories': dataset['categories'],
        'images': [],
        'annotations': []
    }
    
    # 复制图像并更新标注
    print(f"处理 {len(indoor_images)} 张图像...")
    
    new_image_id = 1
    image_id_map = {}  # 原始ID到新ID的映射
    
    success_count = 0
    error_count = 0
    
    for item in tqdm(indoor_images, desc="复制图像"):
        img = item['image']
        annotations = item['annotations']
        
        # 获取图像ID
        image_id = img['id']
        
        # 尝试查找图像文件
        original_filename = img.get('file_name')
        image_path = find_image_file(image_id, args.image_dir, original_filename)
        
        if not image_path:
            error_count += 1
            if error_count <= 10:  # 只显示前10个错误
                print(f"无法找到图像ID {image_id} 的文件")
            continue
        
        # 复制图像
        new_filename = f"indoor_{new_image_id:06d}.jpg"
        new_image_path = os.path.join(images_dir, new_filename)
        
        try:
            # 确保图像是JPEG格式
            img_data = Image.open(image_path)
            img_data = img_data.convert('RGB')
            img_data.save(new_image_path, 'JPEG')
            success_count += 1
        except Exception as e:
            print(f"处理图像 {image_path} 失败: {e}")
            continue
            
        # 创建新的图像记录
        new_img = img.copy()
        new_img['id'] = new_image_id
        new_img['file_name'] = new_filename
        new_img['width'] = img_data.width
        new_img['height'] = img_data.height
        
        # 更新图像ID映射
        image_id_map[image_id] = new_image_id
        
        # 添加到新数据集
        new_dataset['images'].append(new_img)
        
        # 处理标注
        for ann in annotations:
            new_ann = ann.copy()
            new_ann['image_id'] = new_image_id
            new_dataset['annotations'].append(new_ann)
        
        new_image_id += 1
    
    # 保存数据集
    if success_count == 0:
        print("未成功处理任何图像，终止")
        return
    
    # 打印统计信息
    print(f"\n成功处理 {success_count}/{len(indoor_images)} 张图像 ({success_count/len(indoor_images)*100:.1f}%)")
    
    # 保存标注文件
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(new_dataset, f)
    
    # 创建类别映射文件
    category_map = {cat['id']: cat['name'] for cat in new_dataset['categories']}
    category_map_path = os.path.join(output_dir, "category_map.json")
    with open(category_map_path, 'w') as f:
        json.dump(category_map, f)
    
    # 分析数据集
    analyze_dataset(new_dataset, category_map, output_dir)
    
    print(f"\n推理数据集创建完成!")
    print(f"数据集目录: {output_dir}")
    print(f"图像目录: {images_dir}")
    print(f"标注文件: {annotations_path}")

def analyze_dataset(dataset, category_map, output_dir):
    """分析数据集并生成统计信息"""
    
    stats = {
        'total_images': len(dataset['images']),
        'total_annotations': len(dataset['annotations']),
        'object_counts': {},
        'image_sizes': []
    }
    
    # 统计物体类别
    for ann in dataset['annotations']:
        cat_id = ann['category_id']
        cat_name = category_map.get(cat_id, f"unknown_{cat_id}")
        stats['object_counts'][cat_name] = stats['object_counts'].get(cat_name, 0) + 1
    
    # 按频率排序
    stats['object_counts'] = dict(sorted(
        stats['object_counts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    # 统计图像尺寸
    for img in dataset['images']:
        if 'width' in img and 'height' in img:
            stats['image_sizes'].append([img['width'], img['height']])
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印统计信息
    print("\n数据集统计:")
    print(f"  - 总图像数: {stats['total_images']}")
    print(f"  - 总标注数: {stats['total_annotations']}")
    
    print("\n室内物体类别分布:")
    # 只显示室内物体类别
    indoor_objects = {name: count for name, count in stats['object_counts'].items() 
                     if name in INDOOR_OBJECTS.values()}
    for i, (name, count) in enumerate(list(indoor_objects.items())[:15]):
        print(f"  - {name}: {count}")

def main():
    parser = argparse.ArgumentParser(description="从LVIS或COCO数据集创建室内场景推理数据集")
    parser.add_argument("--input", "-i", required=True, help="输入的COCO/LVIS格式数据集JSON文件路径")
    parser.add_argument("--image-dir", "-d", required=True, help="原始图像所在目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--min-objects", "-m", type=int, default=3, help="最少室内物体数量")
    parser.add_argument("--min-types", "-t", type=int, default=2, help="最少不同类型的室内物体数量")
    parser.add_argument("--threshold", "-th", type=float, default=0.6, help="室内场景置信度阈值(0-1)")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.7, help="标注置信度阈值")
    parser.add_argument("--max-images", "-n", type=int, default=200, help="最大图像数量，-1表示不限制")
    
    args = parser.parse_args()
    create_inference_dataset(args)

if __name__ == "__main__":
    main()