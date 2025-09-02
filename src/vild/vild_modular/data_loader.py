# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 数据加载模块
"""

import os
import json
import random
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def load_coco_indoor(coco_path, image_root):
    """加载COCO数据集中的室内场景数据"""
    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"注释文件不存在: {coco_path}")
        
    print(f"正在加载数据集: {coco_path}")
    try:
        with open(coco_path, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        raise
    
    # 构建类别映射
    categories = {cat['id']: cat for cat in dataset['categories']}
    
    # 处理图像和标注
    image_dict = {}
    for image in dataset['images']:
        file_name = None
        
        if 'file_name' in image:
            file_name = image['file_name']
        elif 'coco_url' in image:
            file_name = os.path.basename(image['coco_url'])
        else:
            continue
        
        image_dict[image['id']] = {
            'file_name': file_name,
            'height': image.get('height', 0),
            'width': image.get('width', 0),
            'annotations': []
        }
    
    # 添加标注信息
    for ann in dataset['annotations']:
        try:
            image_id = ann['image_id']
            if image_id in image_dict:
                if 'bbox' in ann and 'category_id' in ann:
                    image_dict[image_id]['annotations'].append({
                        'bbox': ann['bbox'],  # [x, y, w, h]
                        'category_id': ann['category_id'],
                        'segmentation': ann.get('segmentation', []),
                        'iscrowd': ann.get('iscrowd', 0)
                    })
        except KeyError:
            continue
    
    # 过滤掉没有标注的图像
    valid_images = [img for img in image_dict.values() if len(img['annotations']) > 0]
    print(f"有效图像数量(含标注): {len(valid_images)}/{len(image_dict)}")
    
    return valid_images, categories

def select_random_test_image(images, image_root, test_index=-1):
    """从数据集中选择一个随机测试图像"""
    if len(images) == 0:
        return None
    
    # 如果指定了测试图像索引，使用它；否则随机选择
    if test_index >= 0 and test_index < len(images):
        img_index = test_index
    else:
        # 随机选择一个图像
        img_index = random.randint(0, len(images) - 1)
    
    img_info = images[img_index]
    img_path = os.path.join(image_root, img_info['file_name'])
    
    if os.path.exists(img_path):
        print(f"📷 选择测试图像: {os.path.basename(img_path)} (索引 {img_index})")
        return img_path
    else:
        print(f"⚠️ 选择的图像不存在: {img_path}")
        return None

class ImprovedCOCOIndoorDataset(Dataset):
    """改进的COCO室内数据集"""
    
    def __init__(self, images_data, image_root, image_size=256, augment=True, max_samples=None):
        self.images_data = images_data
        self.image_root = image_root
        self.image_size = image_size
        self.augment = augment
        
        # 过滤有效图像
        self.valid_images = []
        for img_info in images_data:
            img_path = os.path.join(image_root, img_info['file_name'])
            if os.path.exists(img_path) and len(img_info['annotations']) > 0:
                # 额外检查图像是否可以正确打开
                try:
                    with Image.open(img_path) as img:
                        if img.width > 0 and img.height > 0:
                            self.valid_images.append(img_info)
                except Exception as e:
                    print(f"⚠️ 图像文件无效，跳过: {img_path}")
        
        # 限制样本数量（如果指定）
        if max_samples and len(self.valid_images) > max_samples:
            self.valid_images = random.sample(self.valid_images, max_samples)
        
        # 基本转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 张量增强 (在ToTensor之后应用)
        self.tensor_augment = None
        if augment:
            self.tensor_augment = transforms.Compose([
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ])
            
        # PIL图像增强 (在ToTensor之前应用)
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomGrayscale(p=0.1)
            ])
        else:
            self.augment_transform = None
        
        print(f"📊 数据集初始化完成: {len(self.valid_images)} 有效图像")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 确保图像是有效的
            if image.width == 0 or image.height == 0:
                raise ValueError(f"图像尺寸无效: {image.width}x{image.height}")
                
            # 对PIL图像应用数据增强
            if self.augment_transform and random.random() > 0.5:
                image = self.augment_transform(image)
            
            # 转换为张量
            image_tensor = self.transform(image)
            
            # 对张量应用额外增强
            if self.tensor_augment and random.random() > 0.5:
                image_tensor = self.tensor_augment(image_tensor)
            
            return {
                'image': image_tensor,
                'image_id': img_info.get('id', idx),
                'annotations': img_info['annotations']
            }
            
        except Exception as e:
            # 返回备用图像
            print(f"⚠️ 图像加载失败 {img_path}: {e}")
            # 创建一个随机噪声图像替代
            random_noise = torch.rand(3, self.image_size, self.image_size) * 0.1
            fallback_image = torch.zeros(3, self.image_size, self.image_size) + random_noise
            # 应用标准化，与正常图像一致
            means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            fallback_image = (fallback_image - means) / stds
            
            return {
                'image': fallback_image,
                'image_id': img_info.get('id', idx),
                'annotations': []
            }

def collate_fn(batch):
    """批处理函数"""
    import torch
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    annotations = [item['annotations'] for item in batch]
    
    return {
        'images': images,
        'image_ids': image_ids,
        'annotations': annotations
    }
