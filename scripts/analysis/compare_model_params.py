#!/usr/bin/env python3
"""
对比各个 RT-DETR 变体的参数量（总参数、可训练参数、模型大小）。
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

try:
    from ultralytics import RTDETR
except ImportError as exc:
    raise SystemExit(f"Ultralytics 导入失败: {exc}") from exc

MODEL_DIR = project_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "rt-detr"
WEIGHTS_ROOT = project_root / "runs" / "detect"

MODELS = [
    ("rtdetr_l", "rtdetr-l.yaml", "RT-DETR-L"),
    ("rtdetr_mnv4", "rtdetr-mnv4-hybrid-m.yaml", "RT-DETR-MNV4"),
    ("rtdetr_mnv4_sea", "rtdetr-mnv4-hybrid-m-sea.yaml", "RT-DETR-MNV4-SEA"),
    ("rtdetr_mnv4_sea_bifpn", "rtdetr-mnv4-hybrid-m-sea-bifpn-lite.yaml", "RT-DETR-MNV4-SEA-BiFPN"),
]


def count_parameters(model_path: str) -> Tuple[int, int]:
    """
    从模型定义加载并统计总参数和可训练参数。
    返回: (total_params, trainable_params)
    """
    try:
        model = RTDETR(model_path)
        model_obj = model.model  # 获取实际模型
        
        total_params = sum(p.numel() for p in model_obj.parameters())
        trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    except Exception as e:
        print(f"  ⚠️  加载失败: {e}")
        return 0, 0


def count_from_checkpoint(weight_path: str) -> Tuple[int, int]:
    """
    从权重文件的 state_dict 统计参数。
    返回: (total_params, checkpoint_params)
    """
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        return total_params, total_params
    except Exception as e:
        print(f"  ⚠️  加载检查点失败: {e}")
        return 0, 0


def get_weight_size(weight_path: str) -> float:
    """获取权重文件大小（MB）。"""
    if Path(weight_path).exists():
        return Path(weight_path).stat().st_size / (1024 * 1024)
    return 0.0


def format_number(num: int) -> str:
    """格式化参数数量为易读形式。"""
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def main():
    print("\n" + "=" * 100)
    print("RT-DETR 变体参数量对比")
    print("=" * 100)
    
    results = []
    
    for model_key, yaml_file, label in MODELS:
        print(f"\n📊 {label}")
        print("-" * 100)
        
        # 从模型定义统计
        model_path = str(MODEL_DIR / yaml_file)
        total, trainable = count_parameters(model_path)
        
        print(f"  模型定义参数:")
        print(f"    总参数数: {format_number(total):>12} ({total:>12,})")
        print(f"    可训练参数: {format_number(trainable):>12} ({trainable:>12,})")
        
        # 查找最佳权重并统计大小
        run_dir = WEIGHTS_ROOT / f"{model_key}_single_bs*"
        best_weights = list(Path(WEIGHTS_ROOT).glob(f"{model_key}_single_bs*/weights/best.pt"))
        
        if best_weights:
            best_pt = str(best_weights[0])
            size_mb = get_weight_size(best_pt)
            print(f"  权重文件:")
            print(f"    路径: {best_pt}")
            print(f"    文件大小: {size_mb:.2f} MB")
            
            results.append({
                "model": label,
                "total_params": total,
                "trainable_params": trainable,
                "weight_size_mb": size_mb,
                "best_pt": best_pt,
            })
        else:
            print(f"  ❌ 未找到训练完的权重文件")
    
    # 总结表格
    if results:
        print("\n" + "=" * 100)
        print("汇总表格")
        print("=" * 100)
        header = f"{'模型':<25}{'总参数数':<15}{'可训练参数':<15}{'权重大小(MB)':<15}"
        print(header)
        print("-" * 100)
        
        for r in results:
            print(
                f"{r['model']:<25}"
                f"{format_number(r['total_params']):<15}"
                f"{format_number(r['trainable_params']):<15}"
                f"{r['weight_size_mb']:<15.2f}"
            )
        
        # 找出最小/最大参数量的模型
        min_model = min(results, key=lambda x: x["total_params"])
        max_model = max(results, key=lambda x: x["total_params"])
        
        print("-" * 100)
        print(f"✅ 最少参数: {min_model['model']} ({format_number(min_model['total_params'])})")
        print(f"✅ 最多参数: {max_model['model']} ({format_number(max_model['total_params'])})")
        print(f"✅ 参数量差异: {format_number(max_model['total_params'] - min_model['total_params'])}")
        
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
