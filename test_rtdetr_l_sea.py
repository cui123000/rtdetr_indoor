#!/usr/bin/env python3
"""
测试 RT-DETR-L-SEA 模型加载
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

print("=" * 70)
print("🧪 测试 RT-DETR-L-SEA 模型")
print("=" * 70)

try:
    print("\n📦 导入 Ultralytics...")
    from ultralytics import RTDETR
    print("✅ 导入成功")
    
    model_path = "/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l-sea.yaml"
    print(f"\n🏗️ 加载模型: {model_path}")
    
    model = RTDETR(model_path)
    print("✅ 模型加载成功")
    
    # 打印模型信息
    print(f"\n📊 模型信息:")
    print(f"   类型: {type(model)}")
    print(f"   模型名称: RT-DETR-L-SEA")
    
    # 统计参数
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
    except:
        print("   无法统计参数")
    
    print("\n✅ RT-DETR-L-SEA 模型测试通过！")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
