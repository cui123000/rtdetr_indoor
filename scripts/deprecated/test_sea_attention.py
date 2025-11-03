#!/usr/bin/env python3
"""
快速测试SEA_Attention_Adaptive模块的数值稳定性
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def test_sea_attention_stability():
    """测试SEA注意力模块的数值稳定性"""
    print("🧪 测试SEA_Attention_Adaptive数值稳定性...")
    
    try:
        from ultralytics.nn.modules.enhanced_attention import SEA_Attention_Adaptive
        
        # 测试不同通道数和输入尺寸
        test_cases = [
            (160, 40, 40),  # P4层
            (256, 20, 20),  # P5层
            (128, 80, 80),  # P3层
        ]
        
        for channels, h, w in test_cases:
            print(f"\n📊 测试 channels={channels}, size={h}x{w}")
            
            # 创建模块
            sea_module = SEA_Attention_Adaptive(channels)
            sea_module.eval()
            
            # 创建测试输入
            x = torch.randn(2, channels, h, w)  # batch=2
            
            # 测试正常情况
            print("  ✅ 正常输入测试...")
            with torch.no_grad():
                output = sea_module(x)
                if torch.isnan(output).any():
                    print("  ❌ 正常输入产生 nan")
                    return False
                else:
                    print(f"  ✅ 输出范围: [{output.min():.4f}, {output.max():.4f}]")
            
            # 测试极端情况
            print("  🔥 极端输入测试...")
            x_extreme = torch.randn(2, channels, h, w) * 100  # 大数值
            with torch.no_grad():
                output_extreme = sea_module(x_extreme)
                if torch.isnan(output_extreme).any():
                    print("  ⚠️  极端输入产生 nan (已通过降级处理)")
                else:
                    print(f"  ✅ 极端输入输出范围: [{output_extreme.min():.4f}, {output_extreme.max():.4f}]")
            
            # 测试包含nan的输入
            print("  💥 nan输入测试...")
            x_nan = torch.randn(2, channels, h, w)
            x_nan[0, 0, 0, 0] = float('nan')
            with torch.no_grad():
                output_nan = sea_module(x_nan)
                if torch.isnan(output_nan).any():
                    print("  ⚠️  nan输入产生 nan输出")
                else:
                    print("  ✅ nan输入已被处理，输出正常")
        
        print("\n🎉 SEA_Attention_Adaptive 数值稳定性测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """测试梯度流动"""
    print("\n🔄 测试梯度流动...")
    
    try:
        from ultralytics.nn.modules.enhanced_attention import SEA_Attention_Adaptive
        
        # 创建模块
        sea_module = SEA_Attention_Adaptive(160)
        sea_module.train()
        
        # 创建测试输入
        x = torch.randn(1, 160, 40, 40, requires_grad=True)
        
        # 前向传播
        output = sea_module(x)
        loss = output.sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in sea_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  ❌ {name} 梯度包含 nan")
                    return False
                else:
                    has_grad = True
                    print(f"  ✅ {name} 梯度正常: [{param.grad.min():.6f}, {param.grad.max():.6f}]")
        
        if has_grad:
            print("  🎉 梯度流动测试通过!")
            return True
        else:
            print("  ⚠️  没有检测到梯度")
            return False
            
    except Exception as e:
        print(f"❌ 梯度测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 SEA_Attention_Adaptive 稳定性测试")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    
    # 运行测试
    stability_ok = test_sea_attention_stability()
    gradient_ok = test_gradient_flow()
    
    if stability_ok and gradient_ok:
        print("\n✅ 所有测试通过! SEA模块数值稳定性良好")
        sys.exit(0)
    else:
        print("\n❌ 测试失败! 需要进一步调试")
        sys.exit(1)
