#!/usr/bin/env python3
"""
RT-DETR 所有版本详细对比分析
包括已训练和待训练的模型
"""

import sys
import csv
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

from ultralytics import RTDETR

# 训练结果目录
RUNS_DIR = project_root / "runs" / "detect"

# 模型配置
MODELS = {
    "RT-DETR-L": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
        "run_dir": "rtdetr_l_single_bs12",
        "status": "已训练",
        "category": "基线",
    },
    "RT-DETR-MNV4": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml",
        "run_dir": "rtdetr_mnv4_single_bs8",
        "status": "已训练",
        "category": "轻量基线",
    },
    "MNV4-SEA": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml",
        "run_dir": "rtdetr_mnv4_sea_single_bs6",
        "status": "已训练",
        "category": "注意力增强",
    },
    "MNV4-SEA-BiFPN": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-bifpn-lite.yaml",
        "run_dir": "rtdetr_mnv4_sea_bifpn_single_bs6",
        "status": "已训练",
        "category": "融合v1",
    },
    "MNV4-SEA-ASFF-v1": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-dysample.yaml",
        "run_dir": "rtdetr_mnv4_sea_asff_single_bs8",
        "status": "已训练",
        "category": "融合v2（失败）",
    },
    "MNV4-SEA-ASFF-v2": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v2.yaml",
        "run_dir": None,
        "status": "待训练",
        "category": "融合v3（完整）",
    },
    "MNV4-SEA-ASFF-v3": {
        "yaml": "ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v3.yaml",
        "run_dir": "rtdetr_mnv4_sea_asff_v3_single_bs8",
        "status": "已训练",
        "category": "融合v4（推荐）",
    },
}


def load_training_results(run_dir):
    """加载训练结果"""
    if run_dir is None:
        return None
    
    results_file = RUNS_DIR / run_dir / "results.csv"
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                return None
            
            # 获取最后一行（最终结果）
            last_row = rows[-1]
            
            # 提取关键指标
            metrics = {
                "mAP50": float(last_row.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(last_row.get("metrics/mAP50-95(B)", 0)),
                "Precision": float(last_row.get("metrics/precision(B)", 0)),
                "Recall": float(last_row.get("metrics/recall(B)", 0)),
            }
            
            # 检查列名变化
            for key in ["mAP50", "mAP50-95", "Precision", "Recall"]:
                if metrics[key] == 0:
                    # 尝试其他可能的列名
                    alt_names = {
                        "mAP50": ["val/box_map50", "box/mAP_50"],
                        "mAP50-95": ["val/box_map", "box/mAP"],
                        "Precision": ["val/box_precision", "box/precision"],
                        "Recall": ["val/box_recall", "box/recall"],
                    }
                    for alt in alt_names.get(key, []):
                        if alt in last_row:
                            metrics[key] = float(last_row[alt])
                            break
            
            return metrics
    except Exception as e:
        print(f"警告: 无法加载 {run_dir} 的结果: {e}")
        return None


def get_model_params(yaml_path):
    """获取模型参数量"""
    try:
        model = RTDETR(yaml_path)
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # 统计ASFF模块
        asff_count = sum(1 for _, m in model.model.named_modules() 
                        if 'ASFF' in type(m).__name__)
        
        return total_params, asff_count
    except Exception as e:
        print(f"警告: 无法加载模型 {yaml_path}: {e}")
        return None, 0


def main():
    print("=" * 100)
    print("RT-DETR 所有版本详细对比分析")
    print("=" * 100)
    print()
    
    # 收集数据
    data = []
    sea_params = None
    sea_map50 = None
    
    for name, config in MODELS.items():
        print(f"📊 分析 {name}...", end=" ")
        
        # 获取参数量
        params, asff_count = get_model_params(config["yaml"])
        
        # 获取训练结果
        results = load_training_results(config["run_dir"])
        
        # 记录SEA基线
        if name == "MNV4-SEA":
            sea_params = params
            if results:
                sea_map50 = results["mAP50"]
        
        row = {
            "模型": name,
            "类别": config["category"],
            "参数量(M)": f"{params/1e6:.2f}" if params else "-",
            "vs SEA": "",
            "ASFF": f"{asff_count}个" if asff_count > 0 else "-",
            "mAP50": f"{results['mAP50']:.4f}" if results else "-",
            "mAP50-95": f"{results['mAP50-95']:.4f}" if results else "-",
            "Precision": f"{results['Precision']:.4f}" if results else "-",
            "Recall": f"{results['Recall']:.4f}" if results else "-",
            "vs SEA性能": "",
            "状态": config["status"],
        }
        
        # 计算与SEA的差异
        if params and sea_params:
            diff = params - sea_params
            diff_pct = (diff / sea_params) * 100
            row["vs SEA"] = f"{diff/1e6:+.2f}M ({diff_pct:+.1f}%)"
        
        # 计算性能差异
        if results and sea_map50:
            perf_diff = results["mAP50"] - sea_map50
            perf_pct = (perf_diff / sea_map50) * 100
            row["vs SEA性能"] = f"{perf_diff:+.4f} ({perf_pct:+.1f}%)"
        
        data.append(row)
        print("✓")
    
    print()
    
    # ============ 输出表格 ============
    print("=" * 100)
    print("📊 模型对比总览")
    print("=" * 100)
    
    # 参数量对比
    print("\n1️⃣  参数量对比")
    print("-" * 100)
    print(f"{'模型':<25} {'类别':<15} {'参数量':<12} {'vs SEA':<20} {'ASFF':<8}")
    print("-" * 100)
    for row in data:
        print(f"{row['模型']:<25} {row['类别']:<15} {row['参数量(M)']:<12} "
              f"{row['vs SEA']:<20} {row['ASFF']:<8}")
    
    # 性能对比
    print("\n2️⃣  性能对比")
    print("-" * 100)
    print(f"{'模型':<25} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} "
          f"{'Recall':<12} {'vs SEA':<20}")
    print("-" * 100)
    for row in data:
        status_icon = "✅" if row["状态"] == "已训练" else "⏳"
        print(f"{status_icon} {row['模型']:<23} {row['mAP50']:<12} {row['mAP50-95']:<12} "
              f"{row['Precision']:<12} {row['Recall']:<12} {row['vs SEA性能']:<20}")
    
    # ============ 关键发现 ============
    print("\n" + "=" * 100)
    print("🔍 关键发现")
    print("=" * 100)
    
    # 找出最佳模型
    trained_models = [row for row in data if row["状态"] == "已训练" and row["mAP50"] != "-"]
    if trained_models:
        best_model = max(trained_models, key=lambda x: float(x["mAP50"]))
        
        print(f"\n✨ 最佳性能: {best_model['模型']}")
        print(f"   • mAP50: {best_model['mAP50']}")
        print(f"   • 参数量: {best_model['参数量(M)']}M")
        print(f"   • 类别: {best_model['类别']}")
    
    # 失败案例分析
    print("\n❌ 失败案例:")
    for row in data:
        if row["状态"] == "已训练" and row["mAP50"] != "-":
            if float(row["mAP50"]) < 0.42:  # 低于预期阈值
                print(f"   • {row['模型']}: mAP50 {row['mAP50']} - {row['类别']}")
    
    # 待训练模型
    print("\n⏳ 待训练模型:")
    for row in data:
        if row["状态"] == "待训练":
            print(f"   • {row['模型']}: {row['参数量(M)']}M - {row['类别']}")
    
    # ============ 推荐策略 ============
    print("\n" + "=" * 100)
    print("💡 训练推荐")
    print("=" * 100)
    
    print("""
优先级1: MNV4-SEA-ASFF-v3 ⭐⭐⭐⭐⭐
  • 参数量: 25.23M (比SEA少13.2%)
  • 特点: 224通道，RepC3×2，完整三尺度ASFF
  • 优势: 参数量减少 + 性能预期稳定
  • 命令: bash train_v3.sh

优先级2: MNV4-SEA-ASFF-v2 ⭐⭐⭐⭐
  • 参数量: 29.78M (比SEA多2.5%)
  • 特点: 256通道，RepC3×3，完整三尺度ASFF
  • 优势: 追求最高性能，完整融合架构
  • 命令: bash train_v2.sh

已验证最佳: MNV4-SEA ⭐⭐⭐⭐⭐
  • 参数量: 29.06M
  • mAP50: 0.4782 (当前最佳)
  • 特点: SEA注意力机制，稳定可靠
""")
    
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
