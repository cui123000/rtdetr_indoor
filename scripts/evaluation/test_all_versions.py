#!/usr/bin/env python3
"""
测试所有RT-DETR MobileNetV4版本
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

# 定义所有版本
VERSIONS = {
    'basic': 'rtdetr-mnv4-basic.yaml',
    'stable': 'rtdetr-mnv4-stable.yaml', 
    'advanced': 'rtdetr-mnv4-advanced.yaml',
    'hybrid': 'rtdetr-mnv4-hybrid-m.yaml'
}

def test_version(name, filename):
    """测试特定版本"""
    try:
        print(f"\n🧪 测试 {name.upper()} 版本...")
        
        from ultralytics import RTDETR
        
        model_path = f"/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{filename}"
        print(f"📄 文件: {filename}")
        
        if not Path(model_path).exists():
            print(f"❌ 配置文件不存在: {model_path}")
            return False
        
        model = RTDETR(model_path)
        
        # 参数统计
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"✅ 加载成功! 参数量: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ {name} 版本失败: {e}")
        return False

def main():
    """测试所有版本"""
    print("🔍 测试所有RT-DETR + MobileNetV4版本")
    print("=" * 50)
    
    results = {}
    
    for name, filename in VERSIONS.items():
        results[name] = test_version(name, filename)
    
    print("\n📊 测试结果汇总:")
    print("=" * 30)
    
    for name, success in results.items():
        status = "✅ 可用" if success else "❌ 失败"
        print(f"{name:<10}: {status}")
    
    working_versions = [name for name, success in results.items() if success]
    
    if working_versions:
        print(f"\n🎉 可用版本: {', '.join(working_versions)}")
        if 'hybrid' in working_versions:
            print("🚀 推荐使用 HYBRID 版本 (最完整的MobileNetV4实现)")
        elif 'advanced' in working_versions:
            print("⭐ 推荐使用 ADVANCED 版本 (平衡性能和稳定性)")
        else:
            print(f"📌 推荐使用 {working_versions[0].upper()} 版本")
    else:
        print("❌ 没有可用版本!")
    
    return len(working_versions) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
