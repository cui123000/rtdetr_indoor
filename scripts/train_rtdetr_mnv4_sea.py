#!/usr/bin/env python3
"""
RT-DETR MobileNetV4 + SEA Attention 训练脚本
优化的SEA注意力机制集成到RT-DETR中，专为室内目标检测设计
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RT-DETR MobileNetV4 + SEA Training')
    
    # Model configuration
    parser.add_argument('--model', type=str, 
                       default='ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml',
                       help='Model configuration file')
    parser.add_argument('--data', type=str, 
                       default='datasets/indoor_enhanced/coco_indoor_enhanced.yaml',
                       help='Dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    # Training configuration
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    
    # Data augmentation
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.1, help='Mixup augmentation probability')
    parser.add_argument('--copy-paste', type=float, default=0.3, help='Copy-paste augmentation probability')
    
    # Device and optimization
    parser.add_argument('--device', type=str, default='0', help='GPU device or cpu')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Enable mixed precision training')
    
    # Output configuration
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='rtdetr_mnv4_sea', help='Experiment name')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Resume and pretrained
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained weights')
    
    # Validation
    parser.add_argument('--val', action='store_true', default=True, help='Validate during training')
    parser.add_argument('--val-period', type=int, default=1, help='Validation period')
    
    return parser.parse_args()

def verify_sea_modules():
    """验证SEA模块是否正确加载"""
    try:
        from ultralytics.nn.modules.sea_attention import (
            Sea_Attention_Simplified,
            OptimizedSEA_Attention,
            TransformerEnhancedSEA,
            create_sea_attention
        )
        print("✅ SEA模块验证成功")
        
        # 测试创建模块
        test_simplified = Sea_Attention_Simplified(64)
        test_optimized = OptimizedSEA_Attention(128, detection_mode=True)
        test_transformer = TransformerEnhancedSEA(256, num_heads=8)
        
        print("✅ SEA模块实例化成功")
        return True
        
    except Exception as e:
        print(f"❌ SEA模块验证失败: {e}")
        return False

def main():
    """主训练函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🚀 RT-DETR MobileNetV4 + 优化SEA注意力训练")
    print("=" * 80)
    print(f"📋 配置信息:")
    print(f"   - 模型配置: {args.model}")
    print(f"   - 数据集: {args.data}")
    print(f"   - 图像尺寸: {args.imgsz}")
    print(f"   - 批次大小: {args.batch_size}")
    print(f"   - 训练轮次: {args.epochs}")
    print(f"   - 设备: {args.device}")
    print(f"   - 优化器: {args.optimizer}")
    
    # 验证SEA模块
    if not verify_sea_modules():
        print("❌ 请先修复SEA模块加载问题")
        return 1
    
    try:
        # 导入RTDETR
        from ultralytics import RTDETR
        
        # 检查模型配置文件
        model_path = Path(args.model)
        if not model_path.exists():
            # 尝试相对路径
            model_path = ROOT / args.model
        
        if not model_path.exists():
            print(f"❌ 模型配置文件不存在: {args.model}")
            return 1
        
        print(f"📄 使用模型配置: {model_path}")
        
        # 创建模型
        if args.resume:
            print(f"📂 从检查点恢复: {args.resume}")
            model = RTDETR(args.resume)
        else:
            print(f"🏗️ 创建新模型")
            model = RTDETR(str(model_path))
            
            # 加载预训练权重
            if args.pretrained and Path(args.pretrained).exists():
                print(f"⚡ 加载预训练权重: {args.pretrained}")
                try:
                    model.load(args.pretrained)
                except Exception as e:
                    print(f"⚠️ 预训练权重加载失败: {e}")
                    print("   继续使用随机初始化权重...")
        
        # 检查数据集配置
        data_path = Path(args.data)
        if not data_path.exists():
            data_path = ROOT / args.data
        
        if not data_path.exists():
            print(f"❌ 数据集配置文件不存在: {args.data}")
            return 1
        
        print(f"📊 使用数据集配置: {data_path}")
        
        # 训练配置
        train_args = {
            'data': str(data_path),
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.imgsz,
            'lr0': args.lr0,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'optimizer': args.optimizer,
            'device': args.device,
            'workers': args.workers,
            'project': args.project,
            'name': args.name,
            'save_period': args.save_period,
            'cache': args.cache,
            'amp': args.mixed_precision,
            'verbose': True,
            'plots': True,
            'save': True,
            'val': args.val,
            'val_period': args.val_period,
            # Data augmentation
            'mosaic': args.mosaic,
            'mixup': args.mixup,
            'copy_paste': args.copy_paste,
        }
        
        print("\n🎯 开始训练...")
        print("-" * 80)
        
        # 开始训练
        results = model.train(**train_args)
        
        print("\n✅ 训练完成!")
        if hasattr(results, 'box'):
            print(f"📊 最佳结果: mAP50={results.box.map50:.4f}, mAP50-95={results.box.map:.4f}")
        
        # 验证模型
        print("\n📈 运行最终验证...")
        val_results = model.val()
        
        if hasattr(val_results, 'box'):
            print(f"🎯 验证结果: mAP50={val_results.box.map50:.4f}, mAP50-95={val_results.box.map:.4f}")
        
        # 导出模型
        save_dir = Path(args.project) / args.name
        best_weights = save_dir / 'weights' / 'best.pt'
        
        if best_weights.exists():
            print(f"\n💾 导出ONNX模型...")
            try:
                model.export(format='onnx', optimize=True, half=True)
                print(f"✅ 模型已导出到: {save_dir / 'weights'}")
            except Exception as e:
                print(f"⚠️ 模型导出失败: {e}")
        
        print(f"\n🎉 训练完成! 结果保存在: {save_dir}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
