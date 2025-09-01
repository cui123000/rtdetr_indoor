#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从LVIS或COCO数据集创建室内场景训练数据集
- 支持多种图像ID映射方式
- 自动查找匹配的图像文件
- 严格的室内场景筛选
- 训练/验证集划分
- 数据增强选项
- 输出标准COCO格式数据集
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
from PIL import ImageEnhance
from pathlib import Path

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

# 核心室内物体 - 这些物体的存在更能确定是室内场景
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

# 明确的室外物体类别ID，这些物体出现则直接排除
OUTDOOR_OBJECTS = {
    19: "cow",             # 牛
    20: "elephant",        # 大象
    21: "bear",            # 熊
    22: "zebra",           # 斑马
    23: "giraffe",         # 长颈鹿
    17: "horse",           # 马
    18: "sheep",           # 羊
    88: "parking meter",   # 停车计时器
    90: "traffic light",   # 交通灯
    95: "bench"            # 公园长椅
}

# 定义室内场景类型
SCENE_TYPES = {
    "living_room": [63, 68, 64, 62, 71],  # 沙发, 电视, 盆栽, 椅子, 遥控器
    "bedroom": [65, 62, 83, 79, 80],      # 床, 椅子, 泰迪熊, 书, 时钟
    "kitchen": [66, 74, 75, 77, 78],      # 餐桌, 微波炉, 烤箱, 水槽, 冰箱
    "bathroom": [67, 77, 84, 85],         # 马桶, 水槽, 吹风机, 牙刷
    "office": [62, 69, 70, 72, 73]        # 椅子, 笔记本, 鼠标, 键盘, 手机
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
        f"COCO_val2017_{image_id:012d}.jpg", # COCO验证集格式
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

def extract_indoor_images(dataset, annotations_by_image_id, min_indoor_objects=3, min_indoor_types=2, min_confidence=0.7):
    """
    从数据集中提取包含室内物体的图像，使用严格的筛选条件
    
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
            
        # 检查是否包含室外物体 - 如果包含则直接排除
        has_outdoor_object = False
        for ann in valid_annotations:
            if ann['category_id'] in OUTDOOR_OBJECTS:
                has_outdoor_object = True
                break
        
        if has_outdoor_object:
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
        
        # 更严格的筛选标准
        is_indoor = (
            indoor_object_count >= min_indoor_objects and 
            len(indoor_object_types) >= min_indoor_types and
            (core_indoor_count > 0 or indoor_object_count >= ambiguous_count * 2) and
            core_indoor_count > 0  # 必须至少有一个核心室内物体
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
            
            # 判断场景类型
            scene_type = classify_scene_type(valid_annotations)
            
            indoor_images.append({
                'image': img,
                'annotations': valid_annotations,
                'indoor_object_count': indoor_object_count,
                'indoor_object_types': indoor_object_types,
                'core_indoor_count': core_indoor_count,
                'confidence_score': confidence_score,
                'scene_type': scene_type
            })
    
    # 按室内场景置信度排序
    indoor_images.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    print(f"找到 {len(indoor_images)} 张高置信度室内场景图像")
    return indoor_images

def classify_scene_type(annotations):
    """根据物体类别判断场景类型"""
    # 统计每种场景类型的得分
    scene_scores = {scene_type: 0 for scene_type in SCENE_TYPES}
    
    # 统计图像中的类别ID
    category_ids = [ann['category_id'] for ann in annotations]
    
    # 计算每种场景类型的得分
    for scene_type, key_objects in SCENE_TYPES.items():
        # 计算关键物体匹配数量
        matches = sum(1 for obj_id in key_objects if obj_id in category_ids)
        scene_scores[scene_type] = matches / len(key_objects) if key_objects else 0
    
    # 返回得分最高的场景类型
    if max(scene_scores.values()) > 0:
        return max(scene_scores.items(), key=lambda x: x[1])[0]
    else:
        return "other"

def apply_augmentation(image, annotations, augmentation_type):
    """
    应用数据增强
    
    Args:
        image: PIL图像
        annotations: 标注列表
        augmentation_type: 增强类型('flip', 'brightness', 'contrast', 'random')
        
    Returns:
        增强后的图像和标注
    """
    width, height = image.size
    
    if augmentation_type == 'random':
        # 随机选择一种增强方式
        augmentation_type = random.choice(['flip', 'brightness', 'contrast'])
    
    if augmentation_type == 'flip':
        # 水平翻转
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 更新边界框坐标
        for ann in annotations:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                ann['bbox'] = [width - x - w, y, w, h]
    
    elif augmentation_type == 'brightness':
        # 亮度调整
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
    
    elif augmentation_type == 'contrast':
        # 对比度调整
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
    
    return image, annotations

def create_training_dataset(args):
    """
    创建室内场景训练数据集
    
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
    
    # 创建训练和验证集图像目录
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
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
    
    # 按场景类型统计图像
    scene_type_counts = {}
    for item in indoor_images:
        scene_type = item['scene_type']
        scene_type_counts[scene_type] = scene_type_counts.get(scene_type, 0) + 1
    
    print("\n按场景类型分布:")
    for scene_type, count in scene_type_counts.items():
        print(f"  - {scene_type}: {count} 张图像")
    
    # 所有图像都用于训练
    train_images = indoor_images
    
    print(f"\n训练集: {len(train_images)} 张图像")
    
    # 创建训练和验证数据集
    train_dataset = {
        'info': dataset.get('info', {}),
        'licenses': dataset.get('licenses', []),
        'categories': dataset['categories'],
        'images': [],
        'annotations': []
    }
    
    # 处理训练集图像
    print("\n处理训练集图像...")
    train_id_map = {}  # 原始ID到新ID的映射
    train_new_id = 1
    train_success = 0
    
    for item in tqdm(train_images, desc="处理训练图像"):
        img = item['image']
        annotations = item['annotations']
        scene_type = item['scene_type']
        
        # 获取图像ID
        image_id = img['id']
        
        # 尝试查找图像文件
        original_filename = img.get('file_name')
        image_path = find_image_file(image_id, args.image_dir, original_filename)
        
        if not image_path:
            continue
        
        try:
            # 加载图像
            img_data = Image.open(image_path).convert('RGB')
            
            # 创建新的图像记录
            new_filename = f"train_{train_new_id:06d}.jpg"
            new_image_path = os.path.join(train_dir, new_filename)
            
            # 保存原始图像
            img_data.save(new_image_path, 'JPEG')
            
            new_img = img.copy()
            new_img['id'] = train_new_id
            new_img['file_name'] = new_filename
            new_img['width'] = img_data.width
            new_img['height'] = img_data.height
            new_img['scene_type'] = scene_type
            
            # 添加到训练数据集
            train_dataset['images'].append(new_img)
            
            # 处理标注
            for ann in annotations:
                new_ann = ann.copy()
                new_ann['image_id'] = train_new_id
                train_dataset['annotations'].append(new_ann)
            
            # 如果需要数据增强
            if args.augment:
                aug_id_offset = 1000000  # 增强图像ID的偏移量
                
                for aug_idx, aug_type in enumerate(['flip', 'brightness', 'contrast']):
                    # 应用数据增强
                    aug_img, aug_anns = apply_augmentation(
                        img_data.copy(), 
                        [ann.copy() for ann in annotations], 
                        aug_type
                    )
                    
                    # 创建增强图像的记录
                    aug_new_id = train_new_id + aug_id_offset + aug_idx
                    aug_filename = f"train_{aug_new_id:06d}_{aug_type}.jpg"
                    aug_image_path = os.path.join(train_dir, aug_filename)
                    
                    # 保存增强图像
                    aug_img.save(aug_image_path, 'JPEG')
                    
                    aug_new_img = new_img.copy()
                    aug_new_img['id'] = aug_new_id
                    aug_new_img['file_name'] = aug_filename
                    
                    # 添加到训练数据集
                    train_dataset['images'].append(aug_new_img)
                    
                    # 处理标注
                    for ann in aug_anns:
                        aug_new_ann = ann.copy()
                        aug_new_ann['image_id'] = aug_new_id
                        train_dataset['annotations'].append(aug_new_ann)
            
            # 更新ID映射
            train_id_map[image_id] = train_new_id
            train_new_id += 1
            train_success += 1
            
        except Exception as e:
            print(f"处理训练图像 {image_path} 失败: {e}")
            continue
    
    # 打印统计信息
    print(f"\n成功处理训练集图像: {train_success}/{len(train_images)} ({train_success/len(train_images)*100:.1f}%)")
    
    if args.augment:
        print(f"应用数据增强，总训练样本: {len(train_dataset['images'])}")
    
    # 保存数据集
    train_json_path = os.path.join(output_dir, "annotations_train.json")
    
    with open(train_json_path, 'w') as f:
        json.dump(train_dataset, f)
    
    # 生成数据集统计信息
    create_train_dataset_stats(train_dataset,  output_dir)
    
    # 创建元数据文件
    metadata = {
        'dataset_name': 'indoor_scene_detection',
        'train_dir': os.path.relpath(train_dir, output_dir),
        'train_json': os.path.basename(train_json_path),
        'scene_types': list(SCENE_TYPES.keys()) + ["other"],
        'indoor_objects': INDOOR_OBJECTS,
        'core_indoor_objects': CORE_INDOOR_OBJECTS
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n训练数据集创建完成!")
    print(f"训练数据集: {train_json_path}")
    print(f"元数据文件: {metadata_path}")

def create_train_dataset_stats(train_dataset, output_dir):
    """生成训练数据集统计信息"""
    
    # 统计类别信息
    categories = {cat['id']: cat['name'] for cat in train_dataset['categories']}
    
    train_stats = {
        'total_images': len(train_dataset['images']),
        'total_annotations': len(train_dataset['annotations']),
        'object_counts': {},
        'scene_types': {}
    }
    
    # 统计训练集类别分布
    for ann in train_dataset['annotations']:
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, f"unknown_{cat_id}")
        train_stats['object_counts'][cat_name] = train_stats['object_counts'].get(cat_name, 0) + 1
    
    # 按频率排序
    train_stats['object_counts'] = dict(sorted(
        train_stats['object_counts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    # 统计场景类型分布
    for img in train_dataset['images']:
        if 'scene_type' in img:
            scene_type = img['scene_type']
            train_stats['scene_types'][scene_type] = train_stats['scene_types'].get(scene_type, 0) + 1
    
    # 计算每个图像的平均标注数
    if train_stats['total_images'] > 0:
        train_stats['avg_annotations_per_image'] = train_stats['total_annotations'] / train_stats['total_images']
    
    # 统计边界框尺寸分布
    train_bbox_areas = []
    
    for ann in train_dataset['annotations']:
        if 'bbox' in ann:
            w, h = ann['bbox'][2], ann['bbox'][3]
            train_bbox_areas.append(w * h)
    
    if train_bbox_areas:
        train_stats['bbox_area_stats'] = {
            'min': min(train_bbox_areas),
            'max': max(train_bbox_areas),
            'mean': sum(train_bbox_areas) / len(train_bbox_areas),
            'median': sorted(train_bbox_areas)[len(train_bbox_areas) // 2]
        }
    
    # 保存统计信息
    stats = {
        'train': train_stats,
        'categories': categories,
        'indoor_objects': {str(k): v for k, v in INDOOR_OBJECTS.items()},
        'core_indoor_objects': {str(k): v for k, v in CORE_INDOOR_OBJECTS.items()}
    }
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印统计信息
    print("\n数据集统计:")
    print(f"训练集: {train_stats['total_images']} 张图像, {train_stats['total_annotations']} 个标注")
    
    print("\n类别分布 (前10个):")
    for i, (name, count) in enumerate(list(train_stats['object_counts'].items())[:10]):
        print(f"  - {name}: {count}")
    
    print("\n场景类型分布:")
    for scene_type, count in train_stats['scene_types'].items():
        print(f"  - {scene_type}: {count}")
    
    # 创建简单的HTML报告
    create_train_html_report(stats, output_dir)

def create_train_html_report(stats, output_dir):
    """创建简单的训练集HTML统计报告"""
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>室内场景训练数据集统计</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>室内场景训练数据集统计</h1>
    
    <h2>数据集概览</h2>
    <table>
        <tr>
            <th>集合</th>
            <th>图像数量</th>
            <th>标注数量</th>
            <th>平均每图标注数</th>
        </tr>
        <tr>
            <td>训练集</td>
            <td>{stats['train']['total_images']}</td>
            <td>{stats['train']['total_annotations']}</td>
            <td>{stats['train'].get('avg_annotations_per_image', 0):.2f}</td>
        </tr>
    </table>
    
    <h2>场景类型分布</h2>
    <table>
        <tr>
            <th>场景类型</th>
            <th>数量</th>
        </tr>
    """
    
    # 添加场景类型统计
    scene_types = stats['train']['scene_types'].keys()
    for scene_type in sorted(scene_types):
        count = stats['train']['scene_types'].get(scene_type, 0)
        html_content += f"""
        <tr>
            <td>{scene_type}</td>
            <td>{count}</td>
        </tr>
        """
    
    html_content += """
    </table>
    
    <h2>对象类别分布 (前15个)</h2>
    <table>
        <tr>
            <th>类别</th>
            <th>数量</th>
        </tr>
    """
    
    # 获取前15个最常见的类别
    top_categories = sorted(stats['train']['object_counts'].items(), 
                           key=lambda x: x[1], reverse=True)[:15]
    
    for cat_name, count in top_categories:
        html_content += f"""
        <tr>
            <td>{cat_name}</td>
            <td>{count}</td>
        </tr>
        """
    
    html_content += """
    </table>
    
    <h2>室内核心物体</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>名称</th>
        </tr>
    """
    
    # 添加核心室内物体
    for obj_id, obj_name in sorted(CORE_INDOOR_OBJECTS.items()):
        html_content += f"""
        <tr>
            <td>{obj_id}</td>
            <td>{obj_name}</td>
        </tr>
        """
    
    html_content += """
    </table>
    
    <h2>边界框统计</h2>
    <table>
        <tr>
            <th>统计量</th>
            <th>值</th>
        </tr>
    """
    
    if 'bbox_area_stats' in stats['train']:
        for stat_name in ['min', 'max', 'mean', 'median']:
            val = stats['train']['bbox_area_stats'].get(stat_name, 0)
            html_content += f"""
            <tr>
                <td>面积 {stat_name}</td>
                <td>{val:.1f}</td>
            </tr>
            """
    
    html_content += """
    </table>

    <h2>注意事项</h2>
    <ul>
        <li>这个数据集专注于室内场景检测</li>
        <li>使用了严格的筛选标准，确保高质量的室内场景</li>
        <li>核心室内物体的存在对确定场景类型非常重要</li>
    </ul>

</body>
</html>
    """
    
    # 保存HTML报告
    html_path = os.path.join(output_dir, "dataset_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {html_path}")

def main():
    parser = argparse.ArgumentParser(description="从LVIS或COCO数据集创建室内场景训练数据集")
    parser.add_argument("--input", "-i", required=True, help="输入的COCO/LVIS格式数据集JSON文件路径")
    parser.add_argument("--image-dir", "-d", required=True, help="原始图像所在目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--min-objects", "-m", type=int, default=3, help="最少室内物体数量")
    parser.add_argument("--min-types", "-t", type=int, default=2, help="最少不同类型的室内物体数量")
    parser.add_argument("--threshold", "-th", type=float, default=0.6, help="室内场景置信度阈值(0-1)")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.7, help="标注置信度阈值")
    parser.add_argument("--max-images", "-n", type=int, default=1000, help="最大图像数量，-1表示不限制")
    parser.add_argument("--augment", "-a", action="store_true", help="是否应用数据增强")
    
    args = parser.parse_args()
    create_training_dataset(args)

if __name__ == "__main__":
    main()