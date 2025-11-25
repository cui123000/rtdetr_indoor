#!/usr/bin/env python3
"""
优化的RT-DETR训练脚本 - 解决内存泄漏和速度问题
"""

import os
import sys
import yaml
import torch
import gc
from pathlib import Path
import psutil
import threading
import time

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def setup_memory_optimization():
    """设置内存优化"""
    print("🔧 配置内存优化设置...")
    
    # CUDA内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行以提高速度
    
    # PyTorch设置
    torch.backends.cudnn.benchmark = True  # 启用cuDNN auto-tuner
    torch.backends.cudnn.deterministic = False  # 允许非确定性操作以提高速度
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32以提高速度
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置线程数
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    if torch.cuda.is_available():
        # 清理初始GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 设置内存分数
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   PyTorch版本: {torch.__version__}")

def monitor_memory():
    """内存监控器（后台线程）"""
    def monitor():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                if allocated > 8.0:  # 如果GPU内存超过8GB，触发清理
                    torch.cuda.empty_cache()
                    gc.collect()
            time.sleep(30)  # 每30秒检查一次
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

def get_optimized_config(model_choice):
    """获取优化的训练配置"""
    
    model_configs = {
        '1': {
            'file': 'rtdetr-l.yaml',
            'name': 'rtdetr_l_optimized',
            'batch': 6,
            'lr0': 0.001,
        },
        '2': {
            'file': 'rtdetr-mnv4-hybrid-m.yaml',
            'name': 'rtdetr_mnv4_hybrid_optimized',
            'batch': 4,
            'lr0': 0.0008,
        },
        '3': {
            'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
            'name': 'rtdetr_mnv4_sea_optimized',
            'batch': 3,
            'lr0': 0.0005,
        },
        '4': {
            'file': 'rtdetr-l-sea.yaml',
            'name': 'rtdetr_l_sea_optimized',
            'batch': 5,
            'lr0': 0.0009,
        },
        '11': {
            'file': 'ert-detr.yaml',
            'name': 'ert_detr_optimized',
            'batch': 8,
            'lr0': 0.0006,
        }
    }
    
    if model_choice not in model_configs:
        raise ValueError(f"无效的模型选择: {model_choice}")
    
    model_config = model_configs[model_choice]
    
    # 基础优化配置
    config = {
        'task': 'detect',
        'mode': 'train',
        'model': f'/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_config["file"]}',
        # 指向数据盘上的数据集YAML
        'data': '/root/autodl-tmp/database/homeobjects/HomeObjects-3K.yaml',
        
        # 核心训练参数 - 优化版本
        'epochs': 100,
        'batch': model_config['batch'],
        'imgsz': 640,
        'patience': 15,
        
        # 性能优化设置
        'device': '0',
        'workers': 2,           # 减少workers避免CPU瓶颈
        'amp': True,            # 混合精度训练
        'cache': False,         # 关闭缓存节省内存
        'rect': True,           # 矩形训练提高效率
        'single_cls': False,    # 多类检测
        
        # 优化器设置
        'optimizer': 'AdamW',
        'lr0': model_config['lr0'],
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,
        
        # 数据增强 - 轻量化设置
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,         # 关闭旋转节省计算
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,          # 关闭剪切节省计算
        'perspective': 0.0,     # 关闭透视变换节省计算
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.0,          # 关闭mixup节省内存
        'copy_paste': 0.0,     # 关闭copy_paste节省内存
        
        # 损失权重
        'box': 7.5,
        'cls': 0.5, 
        'dfl': 1.5,
        
        # 验证设置
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        
        # 保存设置
        'save': True,
        'save_period': 10,      # 减少保存频率
        'project': 'runs/detect',
        'name': model_config['name'],
        'exist_ok': True,
        
        # 其他优化设置
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'plots': True,
        'close_mosaic': 10,
        
        # 内存优化专用设置
        'overlap_mask': False,  # 关闭重叠mask节省内存
        'mask_ratio': 4,        # 减少mask比例
    }
    
    return config

def cleanup_memory():
    """强制清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def train_model(model_choice):
    """训练模型"""
    try:
        # 设置环境
        setup_memory_optimization()
        monitor_memory()
        
        # 导入ultralytics
        print("📦 导入Ultralytics...")
        from ultralytics import RTDETR
        
        # 获取配置
        config = get_optimized_config(model_choice)
        
        print(f"\n🚀 开始训练模型: {config['name']}")
        print(f"📄 配置文件: {config['model']}")
        print(f"📊 批次大小: {config['batch']}")
        print(f"🎯 学习率: {config['lr0']}")
        print("=" * 60)
        
        # 创建模型
        model = RTDETR(config['model'])
        
        # 开始训练
        results = model.train(**{k: v for k, v in config.items() if k not in ['model']})
        
        print("\n✅ 训练完成!")
        print(f"📊 最佳mAP50: {results.best_fitness}")
        
        # 清理内存
        cleanup_memory()
        
        return results
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        cleanup_memory()
        raise

def main():
    """主函数"""
    print("🏃‍♂️ RT-DETR 优化训练脚本")
    print("=" * 50)
    
    print("\n📋 可用模型:")
    print("1. RT-DETR-L (原始)")
    print("2. RT-DETR + MobileNetV4")  
    print("3. RT-DETR + MobileNetV4 + SEA")
    
    while True:
        try:
            choice = input("\n请选择模型 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            else:
                print("❌ 请输入 1, 2 或 3")
        except KeyboardInterrupt:
            print("\n👋 退出训练")
            return
    
    try:
        results = train_model(choice)
        print(f"\n🎉 训练成功完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        cleanup_memory()
    except Exception as e:
        print(f"\n💥 训练失败: {e}")
        cleanup_memory()

if __name__ == "__main__":
    main()
