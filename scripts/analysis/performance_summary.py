#!/usr/bin/env python3
"""
生成各版本性能对比可视化报告
"""

# 训练结果汇总
results = {
    "RT-DETR-L": {"params": 32.97, "map50": 0.3144, "map50_95": 0.2137, "category": "基线"},
    "RT-DETR-MNV4": {"params": 24.98, "map50": 0.3990, "map50_95": 0.2684, "category": "轻量基线"},
    "MNV4-SEA": {"params": 29.06, "map50": 0.4566, "map50_95": 0.2973, "category": "注意力增强"},
    "MNV4-SEA-BiFPN": {"params": 29.26, "map50": 0.4167, "map50_95": 0.2813, "category": "融合v1失败"},
    "MNV4-SEA-ASFF-v1": {"params": 27.75, "map50": 0.3889, "map50_95": 0.2513, "category": "融合v2失败"},
    "MNV4-SEA-ASFF-v3": {"params": 25.23, "map50": 0.3593, "map50_95": 0.2339, "category": "融合v3失败"},
}

print("=" * 80)
print("RT-DETR 各版本性能对比总结")
print("=" * 80)
print()

# 找出最佳和最差
best_model = max(results.items(), key=lambda x: x[1]["map50"])
worst_model = min(results.items(), key=lambda x: x[1]["map50"])

print(f"🏆 最佳模型: {best_model[0]}")
print(f"   • mAP50: {best_model[1]['map50']:.4f}")
print(f"   • 参数量: {best_model[1]['params']:.2f}M")
print(f"   • 类别: {best_model[1]['category']}")
print()

print(f"❌ 最差模型: {worst_model[0]}")
print(f"   • mAP50: {worst_model[1]['map50']:.4f}")
print(f"   • 参数量: {worst_model[1]['params']:.2f}M")
print(f"   • 类别: {worst_model[1]['category']}")
print()

# 性能排序
print("=" * 80)
print("📊 性能排名 (按mAP50)")
print("=" * 80)
print()

sorted_models = sorted(results.items(), key=lambda x: x[1]["map50"], reverse=True)

sea_map50 = results["MNV4-SEA"]["map50"]

for rank, (name, data) in enumerate(sorted_models, 1):
    diff = data["map50"] - sea_map50
    diff_pct = (diff / sea_map50) * 100
    
    if rank == 1:
        icon = "🥇"
    elif rank == 2:
        icon = "🥈"
    elif rank == 3:
        icon = "🥉"
    else:
        icon = "  "
    
    status = "✅" if diff >= -0.01 else "❌"
    
    print(f"{icon} #{rank} {name:<25} mAP50: {data['map50']:.4f}  "
          f"({data['params']:.2f}M)  {status}")
    if name != "MNV4-SEA":
        print(f"      vs SEA: {diff:+.4f} ({diff_pct:+.1f}%)")

print()

# 分类汇总
print("=" * 80)
print("📋 分类汇总")
print("=" * 80)
print()

categories = {}
for name, data in results.items():
    cat = data["category"]
    if cat not in categories:
        categories[cat] = []
    categories[cat].append((name, data))

for cat, models in categories.items():
    print(f"【{cat}】")
    for name, data in models:
        print(f"  • {name}: mAP50 {data['map50']:.4f}, {data['params']:.2f}M")
    print()

# 关键结论
print("=" * 80)
print("🔍 关键结论")
print("=" * 80)
print()

print("✅ 成功:")
print("  • MNV4-SEA 是唯一成功的改进，mAP50达到0.4566")
print("  • 相比基线RT-DETR-L (+45.2%)和MNV4 (+14.4%)都有显著提升")
print()

print("❌ 失败:")
print("  • 所有融合网络尝试(BiFPN, ASFF)均失败")
print("  • BiFPN: -8.7%")
print("  • ASFF v1: -14.8%") 
print("  • ASFF v3: -21.3% (最差)")
print()

print("💡 教训:")
print("  • RT-DETR的Transformer解码器已有强大的多尺度融合能力")
print("  • 额外添加融合网络反而破坏了原有平衡")
print("  • 轻量化需谨慎，过度减少通道会严重损害性能")
print()

print("🎯 建议:")
print("  • 使用MNV4-SEA作为最终模型 (mAP50: 0.4566)")
print("  • 停止ASFF方向的尝试")
print("  • 考虑知识蒸馏、数据增强等其他优化方向")
print()

print("=" * 80)
