#!/usr/bin/env python3
"""
简化的RT-DETR训练脚本 - 修复版本
"""

import os
import sys
import torch
import gc
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def setup_environment():
    """设置训练环境"""
    print("🔧 设置训练环境...")
    
    # CUDA内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # PyTorch设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.85)
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

def get_basic_config(model_choice):
    """获取基础训练配置"""
    
    model_configs = {
        '1': {
            'file': 'rtdetr-l.yaml',
            'name': 'rtdetr_l_fixed',
            'batch': 4,
            'lr0': 0.001,
        },
        '2': {
            'file': 'rtdetr-mnv4-hybrid-m.yaml', 
            'name': 'rtdetr_mnv4_fixed',
            'batch': 3,
            'lr0': 0.0008,
        },
        '3': {
            'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
            'name': 'rtdetr_mnv4_sea_fixed',
            'batch': 2,  # 最小batch以避免内存问题
            'lr0': 0.0005,
        }
    }
    
    model_config = model_configs[model_choice]
    
    # 简化的配置，只包含确认有效的参数
    config = {
        'task': 'detect',
        'mode': 'train',
        'model': f'/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_config["file"]}',
        'data': '/home/cui/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml',
        
        # 基本训练参数
        'epochs': 100,
        'batch': model_config['batch'],
        'imgsz': 640,
        'patience': 20,
        
        # 设备设置
        'device': 0,
        'workers': 2,
        'amp': True,
        'cache': False,
        
        # 优化器
        'optimizer': 'AdamW',
        'lr0': model_config['lr0'],
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'cos_lr': True,
        
        # 数据增强 - 最小化设置
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0,
        'perspective': 0,
        'flipud': 0,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0,
        'copy_paste': 0,
        
        # 损失权重
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 验证
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        
        # 保存
        'save': True,
        'save_period': -1,  # 使用 -1 而不是正数
        'project': 'runs/detect',
        'name': model_config['name'],
        'exist_ok': True,
        
        # 其他
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'plots': True,
        'close_mosaic': 10,
    }
    
    return config

def train_model(model_choice):
    """训练模型"""
    try:
        setup_environment()
        
        print("📦 导入RTDETR...")
        from ultralytics import RTDETR
        
        config = get_basic_config(model_choice)
        
        print(f"\n🚀 开始训练: {config['name']}")
        print(f"📄 模型: {config['model']}")
        print(f"📊 Batch: {config['batch']}")
        print(f"🎯 LR: {config['lr0']}")
        print("=" * 50)
        
        # 创建模型
        model = RTDETR(config['model'])
        
        # 开始训练
        results = model.train(**{k: v for k, v in config.items() if k != 'model'})
        
        print("✅ 训练完成!")
        return results
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise

def main():
    """主函数"""
    print("🚀 RT-DETR 训练脚本 (修复版)")
    print("=" * 40)
    
    print("\n选择模型:")
    print("1. RT-DETR-L")
    print("2. RT-DETR + MobileNetV4") 
    print("3. RT-DETR + MobileNetV4 + SEA")
    
    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("❌ 请输入 1, 2 或 3")
        except KeyboardInterrupt:
            print("\n👋 退出")
            return
    
    try:
        train_model(choice)
    except KeyboardInterrupt:
        print("\n⏹️ 训练中断")
    except Exception as e:
        print(f"💥 训练失败: {e}")

if __name__ == "__main__":
    main()
