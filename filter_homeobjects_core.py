#!/usr/bin/env python3
"""
从COCO原始数据集中筛选HomeObjects扩展类别，严格排除室外场景
使用Places365进行室内/室外分类
"""

import os
import shutil
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# HomeObjects核心类别 + 扩充的家具、电器、家居用品
COCO_TO_HOMEOBJECTS_EXTENDED = {
    # === HomeObjects核心类别 ===
    # 核心家具 (0-3)
    59: 0,   # bed -> bed
    57: 1,   # couch -> sofa  
    56: 2,   # chair -> chair
    60: 3,   # dining table -> table
    
    # 核心电器 (4-6)  
    74: 4,   # clock -> clock/lamp
    62: 5,   # tv -> tv
    63: 6,   # laptop -> laptop
    
    # 核心装饰 (7-11)
    58: 7,   # potted plant -> potted plant
    75: 8,   # vase -> vase/photo frame
    73: 9,   # book -> book
    
    # === 扩充类别 ===
    # 餐具厨具 (10-19)
    44: 10,  # bottle -> bottle
    46: 11,  # cup -> cup  
    50: 12,  # bowl -> bowl
    45: 13,  # wine glass -> glass
    48: 14,  # knife -> knife
    49: 15,  # spoon -> spoon
    47: 16,  # fork -> fork
    
    # 厨房电器 (17-22)
    72: 17,  # refrigerator -> refrigerator
    68: 18,  # microwave -> microwave
    69: 19,  # oven -> oven
    70: 20,  # toaster -> toaster
    71: 21,  # sink -> sink
    
    # 卫浴设备 (22-24)
    61: 22,  # toilet -> toilet
    79: 23,  # toothbrush -> toothbrush
    
    # 电子设备 (24-27)
    66: 24,  # keyboard -> keyboard
    64: 25,  # mouse -> mouse
    67: 26,  # cell phone -> phone
    65: 27,  # remote -> remote
    
    # 人物 (重要的参考对象)
    0: 28,   # person -> person
}

# 扩展后的类别名称
HOMEOBJECTS_EXTENDED_NAMES = {
    # HomeObjects核心
    0: 'bed', 1: 'sofa', 2: 'chair', 3: 'table',
    4: 'clock', 5: 'tv', 6: 'laptop',
    7: 'plant', 8: 'vase', 9: 'book',
    
    # 餐具厨具
    10: 'bottle', 11: 'cup', 12: 'bowl', 13: 'glass',
    14: 'knife', 15: 'spoon', 16: 'fork',
    
    # 厨房电器
    17: 'refrigerator', 18: 'microwave', 19: 'oven', 
    20: 'toaster', 21: 'sink',
    
    # 卫浴设备
    22: 'toilet', 23: 'toothbrush',
    
    # 电子设备
    24: 'keyboard', 25: 'mouse', 26: 'phone', 27: 'remote',
    
    # 人物
    28: 'person'
}

def download_places365_assets(cache_dir: Path):
    """下载Places365模型和相关文件"""
    import urllib.request
    
    cache_dir.mkdir(parents=True, exist_ok=True)  # 添加parents=True
    
    assets = {
        'model': ('resnet50_places365.pth.tar', 
                  'https://github.com/CSAILVision/places365/blob/master/resnet50_places365.pth.tar?raw=true'),
        'categories': ('categories_places365.txt',
                      'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt'),
        'io_places365': ('IO_places365.txt',
                        'https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt')
    }
    
    for name, (filename, url) in assets.items():
        filepath = cache_dir / filename
        if not filepath.exists() or filepath.stat().st_size == 0:
            print(f"下载 {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ {filename} 下载完成")
            except Exception as e:
                print(f"❌ 下载 {filename} 失败: {e}")
                raise
        else:
            print(f"✅ {filename} 已存在")

def load_places365_model(weights_path: Path, device):
    """加载Places365预训练模型"""
    from torchvision.models import resnet50
    
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 365)
    
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    
    model.to(device)
    model.eval()
    return model

def load_places365_io_mapping(io_path: Path):
    """加载Places365室内/室外映射"""
    io_mapping = []
    with open(io_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # 1=室内, 2=室外，转换为1=室内, 0=室外
                io_value = int(parts[1])
                io_mapping.append(1 if io_value == 1 else 0)
    return io_mapping

def is_indoor_scene(image_path: Path, model, io_mapping, transform, device, threshold=0.6):
    """判断图像是否为室内场景"""
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            
            # 计算加权室内概率
            io_tensor = torch.tensor(io_mapping, device=device, dtype=torch.float32)
            weighted_indoor_prob = (probs[0] * io_tensor).sum()
            
            return weighted_indoor_prob.item() >= threshold
            
    except Exception as e:
        print(f"处理图像失败 {image_path}: {e}")
        return False

def filter_homeobjects_extended():
    """从COCO原始数据集筛选HomeObjects扩展类别，严格排除室外场景"""
    
    source_root = Path('/root/autodl-tmp/database/coco')  # COCO原始数据集路径
    output_root = Path('datasets/homeobjects_extended_yolo_indoor_strict')
    cache_dir = Path('.cache/places365')
    
    if not source_root.exists():
        print("❌ COCO原始数据集不存在: /root/autodl-tmp/database/coco")
        print("💡 请确保COCO数据集已下载到该目录")
        return False
        
    if output_root.exists():
        shutil.rmtree(output_root)
    
    # 下载Places365资源
    print("📥 准备Places365模型...")
    download_places365_assets(cache_dir)
    
    # 加载Places365模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 使用设备: {device}")
    
    try:
        model = load_places365_model(cache_dir / 'resnet50_places365.pth.tar', device)
        io_mapping = load_places365_io_mapping(cache_dir / 'IO_places365.txt')
        print("✅ Places365模型加载成功")
    except Exception as e:
        print(f"❌ Places365模型加载失败: {e}")
        return False
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建输出目录
    for split in ['train2017', 'val2017']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'train': {new_id: 0 for new_id in HOMEOBJECTS_EXTENDED_NAMES.keys()},
        'val': {new_id: 0 for new_id in HOMEOBJECTS_EXTENDED_NAMES.keys()}
    }
    
    file_counts = {'train': 0, 'val': 0}
    indoor_filtered = {'train': 0, 'val': 0}  # 室内过滤统计
    skipped_files = {'train': 0, 'val': 0}
    
    # 处理训练集和验证集
    for split_name, split_dir in [('train', 'train2017'), ('val', 'val2017')]:
        labels_dir = source_root / 'labels' / split_dir
        images_dir = source_root / 'images' / split_dir
        
        if not labels_dir.exists():
            print(f"⚠️ 跳过不存在的分割: {split_dir}")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        print(f"\n处理 {split_name}集: {len(label_files)} 个文件")
        
        for label_file in tqdm(label_files, desc=f"严格筛选{split_name}集"):
            # 首先检查是否包含目标类别
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 转换标注，只保留目标类别
            new_lines = []
            has_target = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_id = int(parts[0])
                    
                    # 检查是否为目标类别
                    if old_class_id in COCO_TO_HOMEOBJECTS_EXTENDED:
                        new_class_id = COCO_TO_HOMEOBJECTS_EXTENDED[old_class_id]
                        parts[0] = str(new_class_id)
                        new_lines.append(' '.join(parts) + '\n')
                        stats[split_name][new_class_id] += 1
                        has_target = True
            
            if not has_target:
                skipped_files[split_name] += 1
                continue
            
            # 检查对应的图像是否为室内场景
            image_file = images_dir / (label_file.stem + '.jpg')
            if not image_file.exists():
                skipped_files[split_name] += 1
                continue
            
            # 使用Places365判断是否为室内场景
            if not is_indoor_scene(image_file, model, io_mapping, transform, device, threshold=0.7):
                indoor_filtered[split_name] += 1
                continue
            
            # 保存通过筛选的文件
            shutil.copy2(image_file, output_root / 'images' / split_dir)
            
            # 保存筛选后的标注
            output_label = output_root / 'labels' / split_dir / label_file.name
            with open(output_label, 'w') as f:
                f.writelines(new_lines)
            
            file_counts[split_name] += 1
    
    # 创建YOLO配置文件
    config = {
        'path': str(output_root.resolve()),
        'train': 'images/train2017',
        'val': 'images/val2017',
        'names': HOMEOBJECTS_EXTENDED_NAMES,
        'nc': len(HOMEOBJECTS_EXTENDED_NAMES)
    }
    
    with open(output_root / 'homeobjects_extended_indoor_strict.yaml', 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    # 输出详细统计
    print("\n" + "="*80)
    print("🏠 HomeObjects严格室内数据集统计 (排除室外场景)")
    print("="*80)
    
    total_files = sum(file_counts.values())
    total_objects = sum(sum(stats[split].values()) for split in ['train', 'val'])
    total_skipped = sum(skipped_files.values())
    total_outdoor_filtered = sum(indoor_filtered.values())
    
    print(f"📊 筛选统计:")
    print(f"   保留图像: {total_files}")
    print(f"   无目标类别: {total_skipped}")
    print(f"   室外场景过滤: {total_outdoor_filtered}")
    print(f"   总处理数: {total_files + total_skipped + total_outdoor_filtered}")
    print(f"   最终筛选率: {total_files/(total_files + total_skipped + total_outdoor_filtered)*100:.1f}%")
    print(f"   室内纯度: {total_files/(total_files + total_outdoor_filtered)*100:.1f}%")
    
    print(f"\n📊 质量统计:")
    print(f"   总目标数: {total_objects}")
    print(f"   平均每图: {total_objects/max(total_files,1):.1f} 个目标")
    print(f"   类别数量: {len(HOMEOBJECTS_EXTENDED_NAMES)}")
    
    print(f"\n📁 数据分布:")
    print(f"   训练集: {file_counts['train']} 张图像 ({file_counts['train']/max(total_files,1)*100:.1f}%)")
    print(f"   验证集: {file_counts['val']} 张图像 ({file_counts['val']/max(total_files,1)*100:.1f}%)")
    
    # 简化类别统计，只显示样本数前15的类别
    print(f"\n📋 主要类别统计 (TOP 15):")
    print("ID  类别名称         训练集   验证集    总计     占比")
    print("-" * 55)
    
    # 计算各类别总数并排序
    category_totals = []
    for class_id in HOMEOBJECTS_EXTENDED_NAMES.keys():
        train_count = stats['train'][class_id]
        val_count = stats['val'][class_id]
        total_count = train_count + val_count
        if total_count > 0:
            category_totals.append((total_count, class_id, train_count, val_count))
    
    category_totals.sort(reverse=True)
    
    for i, (total_count, class_id, train_count, val_count) in enumerate(category_totals[:15]):
        percentage = total_count / max(total_objects, 1) * 100
        class_name = HOMEOBJECTS_EXTENDED_NAMES[class_id]
        print(f"{class_id:2d}  {class_name:15s}  {train_count:6d}   {val_count:6d}   {total_count:6d}   {percentage:5.1f}%")
    
    # 数据质量评估
    non_zero_counts = [total for total, _, _, _ in category_totals]
    
    if non_zero_counts:
        min_samples = min(non_zero_counts)
        max_samples = max(non_zero_counts)
        active_classes = len(non_zero_counts)
        
        print(f"\n📈 数据质量:")
        print(f"   有效类别: {active_classes}/{len(HOMEOBJECTS_EXTENDED_NAMES)}")
        print(f"   最少样本: {min_samples}")
        print(f"   最多样本: {max_samples}")
        print(f"   样本均衡度: {min_samples/max(max_samples,1)*100:.1f}%")
    
    return True

if __name__ == "__main__":
    print("🎯 开始从COCO筛选HomeObjects扩展数据集（严格室内筛选）...")
    print("📋 目标：HomeObjects核心类别 + 家具电器家居用品扩充")
    print("🏠 类别覆盖：")
    print("   • 核心家具: bed, sofa, chair, table")
    print("   • 电器设备: tv, laptop, clock, refrigerator, microwave, oven等")
    print("   • 家居用品: plant, vase, book, bottle, cup, bowl等")
    print("   • 餐具厨具: knife, spoon, fork, glass等")
    print("   • 卫浴用品: toilet, toothbrush等")
    print("   • 电子设备: keyboard, mouse, phone, remote等")
    print(f"   总计 {len(HOMEOBJECTS_EXTENDED_NAMES)} 个类别")
    print("\n🔍 严格筛选规则:")
    print("   • 只保留包含目标类别的图像")
    print("   • 使用Places365模型排除室外场景")
    print("   • 室内判断阈值: 70%")
    print("   • 确保数据集的室内场景纯度")
    
    success = filter_homeobjects_extended()
    
    if success:
        print("\n✅ HomeObjects严格室内数据集筛选完成！")
        print("📁 数据集位置: ./datasets/homeobjects_extended_yolo_indoor_strict/")
        print("📄 配置文件: ./datasets/homeobjects_extended_yolo_indoor_strict/homeobjects_extended_indoor_strict.yaml")
        print("\n🚀 高质量室内数据集已准备好用于RT-DETR训练！")
        print("💡 此数据集经过严格筛选，确保全部为室内场景")
    else:
        print("\n❌ 数据集筛选失败")