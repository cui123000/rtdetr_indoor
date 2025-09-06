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
    'vild_temperature': 0.07,  # 增加温度参数，增强鲁棒性
    'vild_similarity_threshold': 0.35,  # 保持相似度阈值
    
    # 投影器模型路径
    'projector_path': os.path.join(CHECKPOINTS_DIR, 'best_model.pth'),
    
    # 防过拟合设置
    'feature_dropout': 0.1,  # 特征层的dropout率
    'use_label_smoothing': True,  # 使用标签平滑
    'label_smoothing': 0.1,  # 标签平滑系数
    'use_mixup': True,  # 使用Mixup数据增强
    'mixup_alpha': 0.2  # Mixup参数
}

# 训练参数 - 针对RTX 4090优化并防止过拟合
TRAINING_CONFIG = {
    'batch_size': 32,  # 减小批量大小，有助于泛化
    'max_epochs': 30,
    'learning_rate': 1e-5,  # 降低学习率，减缓训练过程
    'weight_decay': 1e-4,  # 增加权重衰减，加强正则化
    'patience': 10,    # 增加早停耐心值
    'image_size': 224,
    'augment': True,
    'augment_strength': 'strong',  # 增加数据增强强度
    'num_workers': 4,  
    'pin_memory': True,
    'use_amp': True,   # 保持自动混合精度
    'gradient_accumulation_steps': 2,  # 增加梯度累积步数，稳定训练
    'max_samples': 40000,  # 保持样本数限制
    'dropout_rate': 0.2,  # 添加dropout率参数
    'early_stopping_metric': 'val_loss',  # 基于验证损失进行早停
    'scheduler': 'cosine',  # 使用余弦退火学习率调度
    'warmup_epochs': 2,  # 添加预热阶段
}

# 推理参数
INFERENCE_CONFIG = {
    'batch_size': 1,
    'image_size': 640,
    'score_threshold': 0.35,  # 降低分数阈值，以捕获更多物体，特别是人物
    'nms_threshold': 0.45    # 保持NMS阈值不变
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
    
    # 动物类别 (扩展更详细的分类)
    'animal': [
        # 常见宠物
        'cat', 'dog', 'bird', 'fish', 'hamster', 'rabbit', 'turtle', 'lizard', 'snake',
        # 猫的详细分类
        'kitten', 'kitty', 'tabby cat', 'domestic cat', 'feline', 'house cat', 
        'pet cat', 'short hair cat', 'long hair cat', 'siamese cat', 'persian cat',
        # 狗的详细分类
        'puppy', 'canine', 'pet dog', 'domestic dog',
        # 其他宠物
        'parrot', 'parakeet', 'goldfish', 'guinea pig', 'gerbil', 'mouse', 'rat',
        'ferret', 'chinchilla', 'hedgehog', 'iguana', 'chameleon', 'frog', 'toad',
        # 一般动物类别
        'animal', 'pet', 'mammal', 'reptile', 'amphibian', 'rodent', 'bird'
    ]
}
