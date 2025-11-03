#!/usr/bin/env python3
"""
RT-DETR 四个版本融合有效性分析
对比四个模型的参数量、精确率、召回率、mAP 等关键指标，评估融合策略的有效性。

分析维度：
  1. 参数量对比 (总参数、可训练参数)
  2. 性能对比 (mAP50, mAP50-95, Precision, Recall)
  3. 效率对比 (参数量vs性能收益)
  4. 融合有效性评分
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

try:
    from ultralytics import RTDETR
except ImportError as exc:
    raise SystemExit(f"Ultralytics 导入失败: {exc}") from exc

# ============ 模型定义 ============
MODEL_DIR = project_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "rt-detr"
RUNS_ROOT = project_root / "runs" / "detect"

MODELS = [
    {
        "key": "rtdetr_l",
        "yaml": "rtdetr-l.yaml",
        "label": "RT-DETR-L",
        "description": "基础大模型（性能上限）",
        "category": "baseline",
    },
    {
        "key": "rtdetr_mnv4",
        "yaml": "rtdetr-mnv4-hybrid-m.yaml",
        "label": "RT-DETR-MNV4",
        "description": "MobileNetV4 混合主干（轻量基线）",
        "category": "lightweight",
    },
    {
        "key": "rtdetr_mnv4_sea",
        "yaml": "rtdetr-mnv4-hybrid-m-sea.yaml",
        "label": "RT-DETR-MNV4-SEA",
        "description": "MNV4 + SEA 注意力（融合v1）",
        "category": "fusion_v1",
    },
    {
        "key": "rtdetr_mnv4_sea_bifpn",
        "yaml": "rtdetr-mnv4-hybrid-m-sea-bifpn-lite.yaml",
        "label": "RT-DETR-MNV4-SEA-BiFPN",
        "description": "MNV4 + SEA + BiFPN-Lite（融合v2）",
        "category": "fusion_v2",
    },
    {
        "key": "rtdetr_mnv4_sea_asff",
        "yaml": "rtdetr-mnv4-hybrid-m-sea-asff-dysample.yaml",
        "label": "RT-DETR-MNV4-SEA-ASFF-v1",
        "description": "MNV4 + SEA + ASFF简化版（融合v3-失败）",
        "category": "fusion_v3_failed",
    },
    {
        "key": "rtdetr_mnv4_sea_asff_v2",
        "yaml": "rtdetr-mnv4-hybrid-m-sea-asff-v2.yaml",
        "label": "RT-DETR-MNV4-SEA-ASFF-v2",
        "description": "MNV4 + SEA + 完整ASFF，256通道（融合v4-完整版）",
        "category": "fusion_v4",
    },
    {
        "key": "rtdetr_mnv4_sea_asff_v3",
        "yaml": "rtdetr-mnv4-hybrid-m-sea-asff-v3.yaml",
        "label": "RT-DETR-MNV4-SEA-ASFF-v3",
        "description": "MNV4 + SEA + 完整ASFF，224通道（融合v5-轻量版）",
        "category": "fusion_v5",
    },
]


def format_number(num: float, is_percent: bool = False) -> str:
    """格式化数字为易读形式。"""
    if is_percent:
        return f"{num:.2f}%"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.0f}"


def count_model_parameters(model_config: Dict) -> Tuple[int, int, float]:
    """
    统计模型参数。
    返回: (总参数, 可训练参数, 参数量MB)
    """
    try:
        model_path = str(MODEL_DIR / model_config["yaml"])
        model = RTDETR(model_path)
        model_obj = model.model
        
        total_params = sum(p.numel() for p in model_obj.parameters())
        trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
        
        # 估计参数大小（每个参数4字节float32）
        param_size_mb = total_params * 4 / (1024 * 1024)
        
        return total_params, trainable_params, param_size_mb
    except Exception as e:
        print(f"  ⚠️  参数统计失败: {e}")
        return 0, 0, 0.0


def load_training_results(model_config: Dict) -> Optional[Dict]:
    """从结果文件中加载训练指标。"""
    try:
        # 查找结果 JSON 文件
        analysis_dir = RUNS_ROOT / "analysis"
        if not analysis_dir.exists():
            return None
        
        # 使用更精确的匹配：模型 key + "_single_" 来区分不同变体
        # rtdetr_mnv4 → rtdetr_mnv4_single_*.json
        # rtdetr_mnv4_sea → rtdetr_mnv4_sea_single_*.json
        # rtdetr_mnv4_sea_bifpn → rtdetr_mnv4_sea_bifpn_single_*.json
        pattern = f"{model_config['key']}_single_*.json"
        result_files = list(analysis_dir.glob(pattern))
        
        if not result_files:
            return None
        
        # 取最新的结果文件
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        metrics = data.get("metrics", {})
        return metrics
    except Exception as e:
        print(f"  ⚠️  加载结果失败: {e}")
        return None


def extract_metrics(metrics_dict: Dict) -> Dict[str, float]:
    """从 metrics 字典中提取关键指标。"""
    result = {}
    
    # 优先级顺序查找关键指标
    metric_keys = {
        "map50": ["metrics/mAP50(B)", "metrics/mAP50", "map50", "metrics/mAP50-0.5"],
        "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map", "metrics/mAP50-0.95"],
        "precision": ["metrics/precision(B)", "metrics/precision", "precision"],
        "recall": ["metrics/recall(B)", "metrics/recall", "recall"],
    }
    
    for key, candidates in metric_keys.items():
        for candidate in candidates:
            if candidate in metrics_dict:
                try:
                    result[key] = float(metrics_dict[candidate])
                    break
                except (ValueError, TypeError):
                    continue
    
    return result


def analyze_fusion_effectiveness() -> Dict:
    """分析四个版本的融合有效性。"""
    results = []
    
    print("\n" + "=" * 120)
    print("RT-DETR 四个版本融合有效性分析")
    print("=" * 120)
    
    for model_config in MODELS:
        print(f"\n📊 {model_config['label']} - {model_config['description']}")
        print("-" * 120)
        
        # 1. 参数量统计
        print(f"  📈 参数统计:")
        total_params, trainable_params, param_size_mb = count_model_parameters(model_config)
        print(f"    总参数数: {format_number(total_params):<15} ({total_params:,})")
        print(f"    可训练参数: {format_number(trainable_params):<15} ({trainable_params:,})")
        print(f"    模型大小(估算): {param_size_mb:.2f} MB")
        
        # 2. 训练结果指标
        print(f"  🎯 性能指标:")
        metrics = load_training_results(model_config)
        
        if metrics:
            key_metrics = extract_metrics(metrics)
            if key_metrics:
                map50 = key_metrics.get("map50", 0.0)
                map50_95 = key_metrics.get("map50_95", 0.0)
                precision = key_metrics.get("precision", 0.0)
                recall = key_metrics.get("recall", 0.0)
                
                print(f"    mAP50: {map50:.4f}")
                print(f"    mAP50-95: {map50_95:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall: {recall:.4f}")
            else:
                print(f"    ❌ 未找到关键指标")
                map50, map50_95, precision, recall = 0.0, 0.0, 0.0, 0.0
        else:
            print(f"    ❌ 未找到训练结果")
            map50, map50_95, precision, recall = 0.0, 0.0, 0.0, 0.0
        
        results.append({
            "model": model_config,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_size_mb": param_size_mb,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
        })
    
    return results


def generate_comparison_table(results: List[Dict]) -> None:
    """生成对比表格。"""
    print("\n" + "=" * 120)
    print("汇总对比表")
    print("=" * 120)
    
    # 表头
    header = (
        f"{'模型':<25} "
        f"{'参数数(M)':<12} "
        f"{'参数大小(MB)':<12} "
        f"{'mAP50':<10} "
        f"{'mAP50-95':<12} "
        f"{'Precision':<12} "
        f"{'Recall':<10}"
    )
    print(header)
    print("-" * 120)
    
    for result in results:
        label = result["model"]["label"]
        total_params_m = result["total_params"] / 1e6
        param_size = result["param_size_mb"]
        map50 = result["map50"]
        map50_95 = result["map50_95"]
        precision = result["precision"]
        recall = result["recall"]
        
        print(
            f"{label:<25} "
            f"{total_params_m:<12.2f} "
            f"{param_size:<12.2f} "
            f"{map50:<10.4f} "
            f"{map50_95:<12.4f} "
            f"{precision:<12.4f} "
            f"{recall:<10.4f}"
        )
    
    print("-" * 120)


def calculate_fusion_score(results: List[Dict]) -> None:
    """计算融合有效性评分。"""
    print("\n" + "=" * 120)
    print("融合有效性评分")
    print("=" * 120)
    
    # 基线：RT-DETR-L
    baseline = next((r for r in results if r["model"]["key"] == "rtdetr_l"), None)
    if not baseline:
        print("❌ 无法找到基线模型 (rtdetr_l)")
        return
    
    baseline_map50 = baseline["map50"]
    baseline_params = baseline["total_params"]
    baseline_precision = baseline.get("precision", 0)
    baseline_recall = baseline.get("recall", 0)
    
    print(f"\n📍 基线: {baseline['model']['label']}")
    print(f"   mAP50: {baseline_map50:.4f}")
    print(f"   参数数: {baseline_params:,}")
    
    # 对比其他模型（融合版本）
    print(f"\n🔍 融合版本评分:")
    print("-" * 120)
    
    fusion_models = [
        ("rtdetr_mnv4", "轻量基线（参考）"),
        ("rtdetr_mnv4_sea", "融合版本 v1（SEA）"),
        ("rtdetr_mnv4_sea_bifpn", "融合版本 v2（SEA+BiFPN）"),
        ("rtdetr_mnv4_sea_asff", "融合版本 v3（SEA+ASFF+DySample）"),
    ]
    
    for model_key, desc in fusion_models:
        result = next((r for r in results if r["model"]["key"] == model_key), None)
        if not result:
            continue
        
        label = result["model"]["label"]
        map50 = result["map50"]
        params = result["total_params"]
        precision = result.get("precision", 0)
        recall = result.get("recall", 0)
        
        # 计算收益（相对于基线L）
        map50_gain = ((map50 - baseline_map50) / baseline_map50 * 100) if baseline_map50 > 0 else 0
        param_increase = ((params - baseline_params) / baseline_params * 100)
        
        # 效率评分 = mAP增益 / 参数增加
        efficiency = (map50_gain / param_increase) if param_increase > 0 else (map50_gain if map50_gain > 0 else 0)
        
        # 综合评分
        score = 5  # 基础分
        
        # mAP增益评分（±40%）
        if map50_gain > 10:
            score += 3
        elif map50_gain > 5:
            score += 2
        elif map50_gain > 0:
            score += 1
        elif map50_gain < 0:
            score -= 2
        
        # 参数效率评分
        if param_increase < -10:
            score += 2  # 参数减少加分
        elif param_increase < 0:
            score += 1
        elif param_increase > 50:
            score -= 2  # 参数增加太多扣分
        elif param_increase > 20:
            score -= 1
        
        # 精确率和召回率
        if precision > baseline_precision:
            score += 0.5
        if recall > baseline_recall:
            score += 0.5
        
        score = max(1, min(score, 10))  # 限制在 1-10
        
        # 判断有效性
        is_effective = map50_gain > 5 or (map50_gain > 0 and param_increase < 0)
        status = "✅" if is_effective else "❌"
        
        print(f"\n{status} {label} - {desc}")
        print(f"   mAP50 增益: {map50_gain:+.2f}% (baseline={baseline_map50:.4f}, 当前={map50:.4f})")
        print(f"   参数变化: {param_increase:+.2f}% (baseline={baseline_params/1e6:.2f}M, 当前={params/1e6:.2f}M)")
        print(f"   Precision: {precision:.4f} (baseline={baseline_precision:.4f})")
        print(f"   Recall: {recall:.4f} (baseline={baseline_recall:.4f})")
        print(f"   效率指数: {efficiency:+.4f} (增益/参数增加)")
        print(f"   综合评分: {score:.1f}/10")


def generate_recommendations(results: List[Dict]) -> None:
    """生成改进建议。"""
    print("\n" + "=" * 120)
    print("建议与总结")
    print("=" * 120)
    
    # 获取基线 L
    baseline_l = next((r for r in results if r["model"]["key"] == "rtdetr_l"), None)
    mnv4 = next((r for r in results if r["model"]["key"] == "rtdetr_mnv4"), None)
    sea_version = next((r for r in results if r["model"]["key"] == "rtdetr_mnv4_sea"), None)
    bifpn_version = next((r for r in results if r["model"]["key"] == "rtdetr_mnv4_sea_bifpn"), None)
    asff_version = next((r for r in results if r["model"]["key"] == "rtdetr_mnv4_sea_asff"), None)
    
    if not baseline_l:
        return
    
    baseline_map50 = baseline_l["map50"]
    baseline_params = baseline_l["total_params"]
    
    # 分析 MNV4 基线（参考）
    if mnv4:
        mnv4_gain = ((mnv4["map50"] - baseline_map50) / baseline_map50 * 100)
        mnv4_param_change = ((mnv4["total_params"] - baseline_params) / baseline_params * 100)
        
        print(f"\n1️⃣ MobileNetV4 基线（参考点）:")
        print(f"   性能: mAP50 {mnv4_gain:+.2f}% ({mnv4['map50']:.4f})")
        print(f"   参数: {mnv4_param_change:+.2f}% ({mnv4['total_params']/1e6:.2f}M)")
        print(f"   → 轻量化基线，作为融合版本的起点")
    
    # 分析 SEA 融合
    if sea_version:
        sea_gain = ((sea_version["map50"] - baseline_map50) / baseline_map50 * 100)
        sea_param_increase = ((sea_version["total_params"] - baseline_params) / baseline_params * 100)
        
        print(f"\n2️⃣ SEA 注意力融合（融合 v1）:")
        print(f"   性能: mAP50 {sea_gain:+.2f}% ({sea_version['map50']:.4f}) vs 基线L({baseline_map50:.4f})")
        print(f"   参数: {sea_param_increase:+.2f}% ({sea_version['total_params']/1e6:.2f}M)")
        
        if sea_gain > mnv4_gain + 5:
            print(f"   ✅ 有效融合：相比MNV4基线性能提升 {sea_gain - ((mnv4['map50'] - baseline_map50) / baseline_map50 * 100):+.2f}%")
            print(f"   → 建议保留并进一步优化 SEA 模块配置")
        elif sea_gain > mnv4_gain:
            print(f"   🟡 轻微改进：相比MNV4基线性能提升 {sea_gain - ((mnv4['map50'] - baseline_map50) / baseline_map50 * 100):+.2f}%")
            print(f"   → 建议减少 SEA 模块数量或采用轻量级版本")
        else:
            print(f"   ❌ 融合无效：相比MNV4基线性能下降 {sea_gain - ((mnv4['map50'] - baseline_map50) / baseline_map50 * 100):.2f}%")
            print(f"   → 需要重新设计融合策略或检查训练配置")
    
    # 分析 BiFPN 融合
    if bifpn_version:
        bifpn_gain = ((bifpn_version["map50"] - baseline_map50) / baseline_map50 * 100)
        bifpn_param_increase = ((bifpn_version["total_params"] - baseline_params) / baseline_params * 100)
        
        print(f"\n3️⃣ BiFPN-Lite 融合（融合 v2）:")
        print(f"   性能: mAP50 {bifpn_gain:+.2f}% ({bifpn_version['map50']:.4f}) vs 基线L({baseline_map50:.4f})")
        print(f"   参数: {bifpn_param_increase:+.2f}% ({bifpn_version['total_params']/1e6:.2f}M)")
        
        if bifpn_gain > mnv4_gain + 5:
            print(f"   ✅ 有效融合：相比MNV4基线性能提升 {bifpn_gain - ((mnv4['map50'] - baseline_map50) / baseline_map50 * 100):+.2f}%")
            print(f"   → 建议在生产环境中使用 BiFPN-Lite 版本")
        elif bifpn_gain > mnv4_gain:
            print(f"   🟡 轻微改进：相比MNV4基线性能提升 {bifpn_gain - ((mnv4['map50'] - baseline_map50) / baseline_map50 * 100):+.2f}%")
            print(f"   → 建议微调 BiFPN 集成策略或融合比例")
        else:
            print(f"   ❌ 融合无效：相比MNV4基线性能下降 {bifpn_gain - ((mnv4['map50'] - baseline_map50) / baseline_map50 * 100):.2f}%")
            print(f"   → 需要重新优化 BiFPN 集成方式")
    
    # 分析 ASFF + DySample 融合
    if asff_version:
        asff_gain = ((asff_version["map50"] - baseline_map50) / baseline_map50 * 100)
        asff_param_change = ((asff_version["total_params"] - baseline_params) / baseline_params * 100)
        
        print(f"\n4️⃣ ASFF + DySample 融合（融合 v3 - 轻量高效）:")
        print(f"   性能: mAP50 {asff_gain:+.2f}% ({asff_version['map50']:.4f}) vs 基线L({baseline_map50:.4f})")
        print(f"   参数: {asff_param_change:+.2f}% ({asff_version['total_params']/1e6:.2f}M)")
        
        # 与SEA对比
        if sea_version:
            sea_map50 = sea_version['map50']
            sea_params = sea_version['total_params']
            asff_vs_sea_perf = ((asff_version['map50'] - sea_map50) / sea_map50 * 100) if sea_map50 > 0 else 0
            asff_vs_sea_param = ((asff_version['total_params'] - sea_params) / sea_params * 100)
            
            print(f"   📊 vs SEA版本:")
            print(f"      性能变化: {asff_vs_sea_perf:+.2f}% ({asff_version['map50']:.4f} vs {sea_map50:.4f})")
            print(f"      参数变化: {asff_vs_sea_param:+.2f}% ({asff_version['total_params']/1e6:.2f}M vs {sea_params/1e6:.2f}M)")
            
            if asff_vs_sea_perf > 2 and asff_vs_sea_param < 0:
                print(f"   ✅ 优秀融合：性能提升且参数减少！")
                print(f"   → 推荐用于生产环境，兼顾性能与效率")
            elif asff_vs_sea_perf > 0 and asff_vs_sea_param < 2:
                print(f"   ✅ 有效融合：性能提升且参数增加很少")
                print(f"   → 建议作为主力模型使用")
            elif asff_vs_sea_perf > 0:
                print(f"   🟡 性能提升但参数增加：需要权衡")
                print(f"   → 可根据应用场景选择")
            else:
                print(f"   ⚠️ 性能未达预期：需要进一步调优")
                print(f"   → 检查训练超参数或增加训练轮数")
        
        # 与BiFPN对比
        if bifpn_version:
            bifpn_map50 = bifpn_version['map50']
            bifpn_params = bifpn_version['total_params']
            asff_vs_bifpn_perf = ((asff_version['map50'] - bifpn_map50) / bifpn_map50 * 100) if bifpn_map50 > 0 else 0
            asff_vs_bifpn_param = ((asff_version['total_params'] - bifpn_params) / bifpn_params * 100)
            
            print(f"   📊 vs BiFPN版本:")
            print(f"      性能变化: {asff_vs_bifpn_perf:+.2f}% ({asff_version['map50']:.4f} vs {bifpn_map50:.4f})")
            print(f"      参数变化: {asff_vs_bifpn_param:+.2f}% ({asff_version['total_params']/1e6:.2f}M vs {bifpn_params/1e6:.2f}M)")
            
            if asff_vs_bifpn_perf > 5:
                print(f"   ✅ ASFF显著优于BiFPN：验证了轻量化融合策略")
            elif asff_vs_bifpn_perf > 0:
                print(f"   ✅ ASFF优于BiFPN：自适应融合更有效")
    
    print(f"\n5️⃣ 后续优化方向:")
    print(f"   • 如果ASFF效果好，可以尝试三尺度ASFF（P3/P4/P5全部使用ASFF）")
    print(f"   • 探索CARAFE上采样替代DySample，可能进一步提升性能")
    print(f"   • 考虑知识蒸馏，用RT-DETR-L指导ASFF版本训练")
    print(f"   • 分析不同尺度特征的融合权重分布，优化通道分配")
    print(f"   • 检查融合模块在 RTX 4090 上的实际推理速度")


def main():
    """主函数。"""
    # 分析融合有效性
    results = analyze_fusion_effectiveness()
    
    # 生成对比表
    generate_comparison_table(results)
    
    # 计算融合评分
    calculate_fusion_score(results)
    
    # 生成建议
    generate_recommendations(results)
    
    print("\n" + "=" * 120 + "\n")


if __name__ == "__main__":
    main()
