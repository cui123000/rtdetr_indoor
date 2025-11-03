#!/usr/bin/env python3
"""
ASFF所有版本参数量和配置对比
"""
import sys
sys.path.insert(0, 'ultralytics')
from ultralytics import RTDETR

print("="*80)
print("ASFF 配置版本完整对比")
print("="*80)

configs = [
    ("SEA基线", "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml", "当前最佳 (mAP50 0.4782)"),
    ("ASFF v1", "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-dysample.yaml", "简化版，已失败"),
    ("ASFF v2", "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v2.yaml", "完整版，256通道"),
    ("ASFF v3", "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v3.yaml", "轻量版，224通道 ⭐推荐"),
    ("v3-lite", "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v3-lite.yaml", "过度轻量，不推荐"),
]

print(f"\n{'版本':<12} {'参数量':<12} {'vs SEA':<12} {'ASFF':<8} {'状态':<20}")
print("-"*80)

sea_params = None
for name, path, desc in configs:
    try:
        model = RTDETR(path)
        params = sum(p.numel() for p in model.model.parameters())
        asff_count = sum(1 for _, m in model.model.named_modules() if 'ASFF' in type(m).__name__)
        
        if sea_params is None:
            sea_params = params
            diff_str = "-"
        else:
            diff = params - sea_params
            diff_pct = (diff / sea_params) * 100
            diff_str = f"{diff/1e6:+.2f}M ({diff_pct:+.1f}%)"
        
        print(f"{name:<12} {params/1e6:>6.2f}M    {diff_str:<12} {asff_count}个     {desc}")
    except Exception as e:
        print(f"{name:<12} 加载失败: {e}")

print("\n" + "="*80)
print("💡 v3-lite 不推荐的原因:")
print("="*80)
print("""
1. 参数量异常过低 (12.38M)
   • 比SEA少了57%，远超预期
   • 甚至低于基础MNV4版本

2. 通道数过小 (192)
   • Backbone输出256通道 → Head降到192通道
   • 降维25%，造成信息瓶颈
   • 无法充分利用backbone特征

3. 特征容量不足
   • ASFF需要足够通道学习自适应权重
   • 192通道对3个尺度融合能力有限
   • 类似v1的过度简化问题

4. 与v1失败风险相似
   • v1: ASFF_Simple + DySample → mAP50 0.3927 (-17.9%)
   • v3-lite: 192通道过窄 → 可能性能更差
""")

print("="*80)
print("✅ 推荐训练顺序:")
print("="*80)
print("""
优先级1: ASFF v3 (25.23M, 224通道) ⭐⭐⭐⭐⭐
  • 参数量符合"减少"要求 (-13.2%)
  • 通道数适度，避免过度轻量
  • 完整三尺度ASFF融合
  • 预期mAP50 > 0.47
  命令: bash train_asff_v3.sh

优先级2: ASFF v2 (29.78M, 256通道) ⭐⭐⭐⭐
  • 追求最高性能
  • 完整256通道 + RepC3×3
  • 参数量稍高 (+2.5%)
  • 预期mAP50 0.48-0.52
  命令: bash train_asff_v2.sh

不推荐: v3-lite (12.38M, 192通道) ❌
  • 通道过窄，信息瓶颈
  • 参数量异常，不合理
  • 高失败风险
""")
print("="*80)
