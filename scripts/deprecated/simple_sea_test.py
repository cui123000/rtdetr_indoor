#!/usr/bin/env python3
"""
简单的SEA模块测试脚本
"""

import torch
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ultralytics"))

def test_sea_modules():
    """测试SEA模块"""
    try:
        from ultralytics.nn.modules.sea_attention import (
            Sea_Attention_Simplified,
            OptimizedSEA_Attention,
        )
        
        print("✅ 模块导入成功")
        
        # 测试简化版
        print("🧪 测试 Sea_Attention_Simplified")
        sea_simple = Sea_Attention_Simplified(64)
        x = torch.randn(2, 64, 32, 32)
        out = sea_simple(x)
        print(f"   输入: {x.shape} -> 输出: {out.shape}")
        print(f"   ✅ 简化版测试通过")
        
        # 测试优化版
        print("🧪 测试 OptimizedSEA_Attention")
        sea_opt = OptimizedSEA_Attention(128, detection_mode=True)
        x = torch.randn(2, 128, 40, 40)
        try:
            out = sea_opt(x)
            print(f"   输入: {x.shape} -> 输出: {out.shape}")
            print(f"   ✅ 优化版测试通过")
        except Exception as e:
            print(f"   ❌ 优化版测试失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 简单SEA模块测试")
    print("=" * 50)
    success = test_sea_modules()
    print("=" * 50)
    if success:
        print("✅ 所有测试通过")
    else:
        print("❌ 测试失败")
