#!/usr/bin/env python3
"""
测试优化后的SEA注意力模块在RT-DETR中的性能表现
"""

import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def test_sea_variants():
    """测试不同SEA变体的性能"""
    print("🧪 测试优化后的SEA注意力模块...")
    
    try:
        from ultralytics.nn.modules.sea_attention import (
            Sea_Attention_Simplified,
            create_sea_attention
        )
        
        # 简化的测试配置 - 只测试简化版本
        test_configs = [
            (64, 32, 32, "Sea_Attention_Simplified", Sea_Attention_Simplified),
            (128, 40, 40, "Sea_Attention_Simplified", Sea_Attention_Simplified),
        ]
        
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        results = {}
        
        for channels, H, W, variant_name, module_class in test_configs:
            print(f"\n📊 测试 {variant_name} - {channels}通道, {H}x{W}")
            
            # 创建模块
            module = module_class(channels)
            module = module.to(device)
            module.eval()
            
            # 创建测试输入
            x = torch.randn(batch_size, channels, H, W).to(device)
            
            # 前向传播测试
            try:
                with torch.no_grad():
                    # 预热
                    for _ in range(3):
                        _ = module(x)
                    
                    # 计时测试
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start_time = time.time()
                    
                    for _ in range(10):
                        output = module(x)
                    
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    avg_time = (time.time() - start_time) / 10
                    
                    # 检查输出
                    if torch.isnan(output).any():
                        print(f"  ❌ 输出包含 nan")
                        continue
                    
                    # 计算参数量
                    params = sum(p.numel() for p in module.parameters())
                    
                    results[variant_name] = {
                        'time_ms': avg_time * 1000,
                        'params': params,
                        'output_shape': output.shape,
                        'memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0
                    }
                    
                    print(f"  ✅ 成功 - 时间: {avg_time*1000:.2f}ms")
                    print(f"  📦 参数量: {params:,}")
                    print(f"  📏 输出形状: {output.shape}")
                    print(f"  🎯 输出范围: [{output.min():.4f}, {output.max():.4f}]")
                    
            except Exception as e:
                print(f"  ❌ 测试失败: {e}")
                continue
        
        # 性能总结
        if results:
            print("\n📈 性能总结:")
            print("=" * 80)
            print(f"{'模块名':<25} {'时间(ms)':<10} {'参数量':<12} {'内存(MB)':<10} {'状态'}")
            print("-" * 80)
            
            for variant_name, metrics in results.items():
                status = "✅ 优秀" if metrics['time_ms'] < 50 else "⚠️ 较慢" if metrics['time_ms'] < 100 else "❌ 慢"
                print(f"{variant_name:<25} {metrics['time_ms']:<10.2f} {metrics['params']:<12,} "
                      f"{metrics['memory_mb']:<10.1f} {status}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_function():
    """测试工厂函数"""
    print("\n🏭 测试SEA注意力工厂函数...")
    
    try:
        from ultralytics.nn.modules.sea_attention import create_sea_attention
        
        test_cases = [
            (64, 'simplified'),
            (128, 'simplified'),
        ]
        
        for dim, variant in test_cases:
            print(f"  📋 创建 {variant} 版本 (dim={dim})")
            module = create_sea_attention(dim, variant=variant, detection_mode=True)
            
            # 简单前向测试
            x = torch.randn(1, dim, 32, 32)
            with torch.no_grad():
                output = module(x)
            
            print(f"    ✅ 输入: {x.shape} -> 输出: {output.shape}")
            
        print("  🎉 工厂函数测试完成!")
        return True
        
    except Exception as e:
        print(f"  ❌ 工厂函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_optimization():
    """测试检测优化功能"""
    print("\n🎯 测试检测优化功能...")
    
    try:
        from ultralytics.nn.modules.sea_attention import Sea_Attention_Simplified
        
        # 测试不同FPN层级
        fpn_configs = [
            (64, 32, 32, "P3层"),
            (128, 40, 40, "P4层"), 
        ]
        
        for channels, H, W, layer_name in fpn_configs:
            print(f"  📊 测试 {layer_name} - {channels}通道, {H}x{W}")
            
            # 创建SEA模块
            sea_module = Sea_Attention_Simplified(channels)
            
            x = torch.randn(2, channels, H, W)
            
            with torch.no_grad():
                output = sea_module(x)
                
                print(f"    ✅ 输出形状: {output.shape}")
                print(f"    ✅ 输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        print("  🎉 检测优化测试完成!")
        return True
        
    except Exception as e:
        print(f"  ❌ 检测优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🤖 优化SEA注意力模块测试 - RT-DETR检测专用")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行测试
    test1 = test_sea_variants()
    test2 = test_factory_function()
    test3 = test_detection_optimization()
    
    if test1 and test2 and test3:
        print("\n✅ 所有测试通过! SEA模块优化成功")
        print("🚀 已准备好集成到RT-DETR训练中")
    else:
        print("\n❌ 部分测试失败! 需要进一步调试")
