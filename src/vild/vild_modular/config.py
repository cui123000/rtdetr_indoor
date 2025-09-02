# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 配置文件
"""

import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

# 数据集路径
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'datasets')
COCO_PATH = os.path.join(DATASET_ROOT, 'coco')
INDOOR_TRAINING_PATH = os.path.join(DATASET_ROOT, 'indoor_training')
INDOOR_INFERENCE_PATH = os.path.join(DATASET_ROOT, 'indoor_inference')
INDOOR_ENHANCED_PATH = os.path.join(DATASET_ROOT, 'indoor_enhanced')

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# 确保目录存在
for d in [OUTPUT_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 模型参数
MODEL_CONFIG = {
    # RT-DETR 参数
    'rtdetr_model_path': os.path.join(PROJECT_ROOT, 'rtdetr-l.pt'),
    'rtdetr_conf_thres': 0.25,
    'rtdetr_iou_thres': 0.45,
    
    # CLIP 参数
    'clip_model_name': 'ViT-B/32',
    
    # ViLD 参数
    'vild_temperature': 0.05,
    'vild_similarity_threshold': 0.35,  # 提高相似度阈值，减少错误检测
    
    # 投影器模型路径
    'projector_path': os.path.join(CHECKPOINTS_DIR, 'best_model.pth')
}

# 训练参数 - 针对RTX 4090优化
TRAINING_CONFIG = {
    'batch_size': 64,  # 增大批量大小，充分利用RTX 4090的24GB内存
    'max_epochs': 25,
    'learning_rate': 2e-5,  # 略微提高学习率以匹配更大的批量
    'weight_decay': 5e-5,
    'patience': 5,
    'image_size': 224,
    'augment': True,
    'num_workers': 4,  # 增加数据加载线程
    'pin_memory': True,
    'use_amp': True,   # 启用自动混合精度
    'gradient_accumulation_steps': 1,  # 梯度累积步数，可根据需要调整
    'max_samples': 20000  # 限制最大样本数，避免内存压力过大
}

# 推理参数
INFERENCE_CONFIG = {
    'batch_size': 1,
    'image_size': 640,
    'score_threshold': 0.5,  # 提高分数阈值，只保留高置信度检测
    'nms_threshold': 0.45    # 略微降低NMS阈值，更严格地抑制重复框
}

# 类别配置
CATEGORIES = {
    # 室内家具类别
    'furniture': [
        "chair", "table", "desk", "bed", "sofa", "couch", "cabinet", 
        "wardrobe", "shelf", "bookshelf", "nightstand", "dresser"
    ],
    
    # 电器类别
    'electronics': [
        "tv", "television", "computer", "laptop", "monitor", "keyboard", 
        "mouse", "refrigerator", "fridge", "microwave", "oven"
    ],
    
    # 厨房用品类别
    'kitchenware': [
        "sink", "stove", "oven", "microwave", "refrigerator", 
        "dishwasher", "bottle", "cup", "glass", "bowl", "plate", "knife"
    ],
    
    # 浴室用品类别
    'bathroom': [
        "toilet", "sink", "bathtub", "shower", "mirror", "towel"
    ],
    
    # 装饰品类别
    'decor': [
        "lamp", "clock", "vase", "plant", "painting", "picture", "curtain", "blind"
    ],
    
    # 文具和书籍
    'stationery': [
        "book", "pen", "pencil", "notebook", "paper"
    ]
}

# 大类别映射 - 将细粒度类别映射到更大的类别组
MACRO_CATEGORIES = {
    # 人物相关
    'person': ['person'],
    
    # 食物相关
    'food': [
        'apple', 'banana', 'orange', 'sandwich', 'carrot', 'broccoli', 
        'hot dog', 'pizza', 'donut', 'cake', 'bowl'
    ],
    
    # 餐具相关
    'tableware': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'plate', 'bowl'],
    
    # 家具类
    'furniture': [
        'chair', 'sofa', 'bed', 'dining table', 'table', 'desk', 'cabinet', 
        'wardrobe', 'shelf', 'bookshelf', 'nightstand', 'dresser', 'bench'
    ],
    
    # 电器类
    'electronics': [
        'tv', 'television', 'laptop', 'computer', 'keyboard', 'mouse', 'remote', 
        'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator'
    ],
    
    # 浴室相关
    'bathroom': ['toilet', 'sink', 'bathtub', 'shower', 'mirror', 'towel'],
    
    # 装饰品
    'decor': ['clock', 'vase', 'plant', 'picture', 'painting', 'curtains'],
}
