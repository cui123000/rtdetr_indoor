#!/usr/bin/env python3
"""
RT-DETR with MobileNetV4 Training Script
使用Ultralytics框架训练RT-DETR with MobileNetV4模型
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def create_training_config():
    """创建训练配置文件"""
    config = {
        # 基本设置
        'task': 'detect',
        'mode': 'train',
        
        # 模型和数据
        'model': '/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml',
        'data': '/home/cui/vild_rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml',
        
        # 训练参数
        'epochs': 100,
        'batch': 4,
        'imgsz': 640,
        'patience': 50,
        
        # 保存设置
        'save': True,
        'save_period': 10,
        'project': str(project_root),
        'name': 'rtdetr_mobilenetv4_indoor',
        'exist_ok': True,
        
        # 优化器和学习率
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # 设备设置
        'device': '0',
        'workers': 4,
        'amp': True,
        
        # 验证设置
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        # 其他设置
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'plots': True,
        'cache': False,
        'pretrained': False,  # 不使用预训练，从头开始训练
    }
    
    return config

def setup_environment():
    """设置训练环境"""
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{project_root}:{project_root}/ultralytics"

def check_model_config():
    """检查模型配置文件是否存在"""
    model_config_path = Path("/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml")
    if not model_config_path.exists():
        print(f"❌ Model config file not found: {model_config_path}")
        return False
    
    print(f"✅ Model config file found: {model_config_path}")
    return True

def check_dataset_config():
    """检查数据集配置文件是否存在"""
    dataset_config_path = Path("/home/cui/vild_rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml")
    if not dataset_config_path.exists():
        print(f"❌ Dataset config file not found: {dataset_config_path}")
        print("Please make sure your dataset is properly configured.")
        return False
    
    print(f"✅ Dataset config file found: {dataset_config_path}")
    return True

def train_model():
    """训练模型"""
    try:
        # 导入Ultralytics YOLO
        from ultralytics import RTDETR
        
        # 创建训练配置
        config = create_training_config()
        
        print("🚀 Starting RT-DETR with MobileNetV4 training...")
        print(f"📊 Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 初始化模型
        print("\n📦 Loading RT-DETR with MobileNetV4 model...")
        model = RTDETR(config['model'])
        
        # 开始训练
        print("\n🏃 Starting training...")
        results = model.train(**config)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config['project']}/{config['name']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 RT-DETR with MobileNetV4 Training Script")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 检查配置文件
    if not check_model_config():
        sys.exit(1)
    
    if not check_dataset_config():
        print("⚠️  Dataset config not found, but continuing with training...")
    
    # 训练模型
    results = train_model()
    
    if results is not None:
        print("\n✅ Training script completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Training script failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
