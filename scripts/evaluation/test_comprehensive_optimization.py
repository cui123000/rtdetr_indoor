#!/usr/bin/env python3
"""
测试综合优化配置
测试策略1+2+4的组合效果
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

from ultralytics import RTDETR

def test_comprehensive_model():
    """测试综合优化模型"""
    
    print("🚀 测试综合优化RT-DETR模型")
    print("=" * 60)
    
    # 模型配置路径
    config_path = "/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-comprehensive-optimized.yaml"
    
    print(f"📋 配置文件: {config_path}")
    
    try:
        # 创建模型
        print("🔧 创建模型...")
        model = RTDETR(config_path)
        
        print(f"✅ 模型创建成功")
        print(f"📊 模型信息:")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 测试前向传播
        print(f"\n🧪 测试前向传播...")
        with torch.no_grad():
            # 创建测试输入
            x = torch.randn(1, 3, 640, 640)
            print(f"📥 输入形状: {x.shape}")
            
            # 前向传播
            try:
                output = model(x)
                print(f"✅ 前向传播成功")
                if isinstance(output, (list, tuple)):
                    print(f"📤 输出数量: {len(output)}")
                    for i, out in enumerate(output):
                        if hasattr(out, 'shape'):
                            print(f"  - 输出{i}: {out.shape}")
                else:
                    print(f"📤 输出形状: {output.shape}")
                    
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                return False
        
        # 显示模型结构（简化）
        print(f"\n🏗️ 模型架构概览:")
        total_params = 0
        for name, module in model.named_modules():
            if any(keyword in name for keyword in ['SEA_Attention_Adaptive', 'FeatureWeightFusion', 'Add']):
                param_count = sum(p.numel() for p in module.parameters())
                total_params += param_count
                print(f"  🎯 {name}: {module.__class__.__name__} ({param_count:,} params)")
        
        print(f"\n📈 优化模块总参数: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        print(f"💡 可能的原因:")
        print(f"  1. 配置文件语法错误")
        print(f"  2. 模块导入失败")
        print(f"  3. 参数配置不匹配")
        return False

def test_training_compatibility():
    """测试训练兼容性"""
    
    print(f"\n🎓 测试训练兼容性...")
    
    try:
        # 数据集配置
        dataset_config = {
            'path': '/home/cui/vild_rtdetr_indoor/datasets/indoor_enhanced',
            'train': 'train',
            'val': 'val',
            'names': {0: 'object'}  # 简化的类别
        }
        
        # 创建临时数据集配置文件
        import yaml
        temp_dataset_path = "/tmp/test_dataset.yaml"
        with open(temp_dataset_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        # 创建模型
        config_path = "/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-comprehensive-optimized.yaml"
        model = RTDETR(config_path)
        
        # 测试训练（1个epoch）
        print(f"🏃 开始测试训练（1个epoch）...")
        
        # 训练配置
        train_args = {
            'data': temp_dataset_path,
            'epochs': 1,
            'batch': 2,  # 小batch size
            'imgsz': 320,  # 小图像尺寸
            'save': False,
            'plots': False,
            'verbose': True,
            'device': 'cpu' if not torch.cuda.is_available() else 'cuda:0'
        }
        
        # 开始训练
        results = model.train(**train_args)
        
        print(f"✅ 训练兼容性测试通过")
        print(f"📊 训练结果预览:")
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.4f}")
        
        # 清理临时文件
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔬 RT-DETR综合优化模型测试")
    print("=" * 80)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU可用: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"⚠️ 使用CPU模式")
    
    # 测试模型
    print(f"\n" + "="*60)
    model_test_passed = test_comprehensive_model()
    
    if model_test_passed:
        print(f"\n" + "="*60)
        training_test_passed = test_training_compatibility()
        
        # 最终结果
        print(f"\n" + "="*80)
        print(f"📋 测试结果总结:")
        print(f"  ✅ 模型构建: {'通过' if model_test_passed else '失败'}")
        print(f"  ✅ 训练兼容: {'通过' if training_test_passed else '失败'}")
        
        if model_test_passed and training_test_passed:
            print(f"\n🎉 综合优化模型测试完全通过!")
            print(f"🚀 可以开始正式训练:")
            print(f"   python scripts/training/train_rtdetr_mobilenetv4_select.py \\")
            print(f"     --config rtdetr-mnv4-comprehensive-optimized.yaml \\")
            print(f"     --epochs 100 --batch 8")
        else:
            print(f"\n❌ 测试失败，需要修复问题")
    
    else:
        print(f"\n❌ 模型构建失败，跳过训练测试")

if __name__ == "__main__":
    main()
