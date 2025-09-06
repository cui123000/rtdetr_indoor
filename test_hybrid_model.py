#!/usr/bin/env python3
"""
快速测试Hybrid版本的RT-DETR MobileNetV4
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def test_hybrid_model():
    """测试hybrid模型"""
    try:
        print("🧪 测试 Hybrid RT-DETR MobileNetV4...")
        
        from ultralytics import RTDETR
        
        model_path = "/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml"
        print(f"📄 加载模型: {model_path}")
        
        model = RTDETR(model_path)
        print("✅ Hybrid模型加载成功!")
        
        # 模型信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 参数量: {total_params:,}")
        
        # 前向传播测试
        import torch
        x = torch.randn(1, 3, 640, 640)
        print(f"🔍 测试前向传播: {x.shape}")
        
        model.model.eval()
        with torch.no_grad():
            output = model.model(x)
        
        print("✅ 前向传播成功!")
        print("🎉 Hybrid版本完全可用!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_model()
    print(f"\n{'🎊 测试成功!' if success else '❌ 测试失败!'}")
    sys.exit(0 if success else 1)
