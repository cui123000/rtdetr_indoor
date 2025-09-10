#!/usr/bin/env python3
"""
RT-DETR MobileNetV4 + SEA Attention 模型验证脚本
验证模型配置是否正确，SEA模块是否正常工作
"""

import torch
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ultralytics"))

def test_model_build():
    """测试模型构建"""
    print("🧪 测试RT-DETR + SEA模型构建...")
    
    try:
        from ultralytics import RTDETR
        
        # 模型配置文件路径
        model_config = ROOT / "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml"
        
        if not model_config.exists():
            print(f"❌ 模型配置文件不存在: {model_config}")
            return False
        
        print(f"📄 使用配置: {model_config}")
        
        # 创建模型
        model = RTDETR(str(model_config))
        print("✅ 模型创建成功")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 创建测试输入
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        print(f"🔍 测试前向传播 - 输入形状: {test_input.shape}")
        print(f"📱 设备: {device}")
        
        with torch.no_grad():
            # 设置为评估模式
            model.eval()
            
            # 前向传播
            output = model(test_input)
            
            print(f"✅ 前向传播成功")
            print(f"📊 输出类型: {type(output)}")
            
            if isinstance(output, (list, tuple)):
                print(f"📏 输出长度: {len(output)}")
                for i, out in enumerate(output):
                    if hasattr(out, 'shape'):
                        print(f"   输出[{i}]形状: {out.shape}")
            elif hasattr(output, 'shape'):
                print(f"📏 输出形状: {output.shape}")
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型统计:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024**2:.2f} MB (FP32)")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sea_integration():
    """测试SEA模块集成"""
    print("\n🔧 测试SEA模块集成...")
    
    try:
        from ultralytics.nn.modules.sea_attention import (
            Sea_Attention_Simplified,
            OptimizedSEA_Attention,
            TransformerEnhancedSEA
        )
        
        # 测试不同的SEA变体
        test_configs = [
            (Sea_Attention_Simplified, 64, "简化版"),
            (OptimizedSEA_Attention, 128, "优化版"),
            (TransformerEnhancedSEA, 256, "Transformer增强版")
        ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for module_class, channels, name in test_configs:
            print(f"  🧩 测试 {name} SEA模块 ({channels}通道)")
            
            # 创建模块
            if module_class == OptimizedSEA_Attention:
                module = module_class(channels, detection_mode=True)
            elif module_class == TransformerEnhancedSEA:
                module = module_class(channels, num_heads=min(8, channels//32))
            else:
                module = module_class(channels)
            
            module = module.to(device)
            module.eval()
            
            # 测试输入
            H, W = 40, 40  # 特征图尺寸
            x = torch.randn(2, channels, H, W).to(device)
            
            with torch.no_grad():
                output = x
                output = module(output)
                
                # 检查输出
                assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
                assert not torch.isnan(output).any(), "输出包含NaN"
                
                print(f"    ✅ {name} 测试通过 - 输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        print("✅ SEA模块集成测试完成")
        return True
        
    except Exception as e:
        print(f"❌ SEA集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n🎯 测试训练兼容性...")
    
    try:
        from ultralytics import RTDETR
        
        model_config = ROOT / "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml"
        model = RTDETR(str(model_config))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()  # 训练模式
        
        # 模拟训练数据
        batch_size = 2
        x = torch.randn(batch_size, 3, 640, 640).to(device)
        
        # 模拟标签 (dummy labels for testing)
        # 这里只是测试前向传播，不测试损失计算
        
        print(f"📝 测试训练模式前向传播...")
        
        # 前向传播
        output = model(x)
        
        print(f"✅ 训练模式前向传播成功")
        
        # 测试梯度计算
        if isinstance(output, (list, tuple)):
            loss = sum(out.sum() for out in output if hasattr(out, 'sum'))
        else:
            loss = output.sum()
        
        loss.backward()
        print("✅ 反向传播成功")
        
        # 检查梯度
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"✅ 梯度计算: {'正常' if has_grad else '异常'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 80)
    print("🤖 RT-DETR MobileNetV4 + SEA Attention 模型验证")
    print("=" * 80)
    
    # 运行测试
    test1 = test_model_build()
    test2 = test_sea_integration() 
    test3 = test_training_compatibility()
    
    print("\n" + "=" * 80)
    print("📋 测试总结:")
    print(f"   🏗️ 模型构建: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"   🔧 SEA集成: {'✅ 通过' if test2 else '❌ 失败'}")
    print(f"   🎯 训练兼容: {'✅ 通过' if test3 else '❌ 失败'}")
    
    if test1 and test2 and test3:
        print("\n🎉 所有测试通过! 模型已准备好进行训练")
        return 0
    else:
        print("\n❌ 部分测试失败! 请检查配置")
        return 1

if __name__ == "__main__":
    exit(main())
