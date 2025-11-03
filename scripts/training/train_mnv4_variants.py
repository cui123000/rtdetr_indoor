#!/usr/bin/env python3
"""
训练单个 RT-DETR 变体 (L、MNV4、MNV4+SEA、MNV4+SEA+BiFPN-Lite)。
在脚本中修改 SELECTED_VARIANT_KEY 以选择要训练的模型。
默认配置针对 RTX 4090 (24 GB) 调整，兼顾显存与吞吐。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ============ 在此处修改要训练的模型 ============
SELECTED_VARIANT_KEY = "rtdetr_mnv4_sea_asff_v3"  # 可选: rtdetr_l, rtdetr_mnv4, rtdetr_mnv4_sea, rtdetr_mnv4_sea_bifpn, rtdetr_mnv4_sea_asff, rtdetr_mnv4_sea_asff_v2
# ============================================

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

try:
    from ultralytics import RTDETR  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(f"Ultralytics 导入失败: {exc}") from exc

DATA_CONFIG = "/home/cui/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml"
MODEL_DIR = project_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "rt-detr"
DEFAULT_PROJECT = "runs/detect"


@dataclass
class VariantConfig:
    key: str
    label: str
    yaml_file: str
    batch: int
    lr0: float
    workers: int
    description: str


VARIANTS = [
    VariantConfig(
        key="rtdetr_l",
        label="RT-DETR-L",
        yaml_file="rtdetr-l.yaml",
        batch=12,
        lr0=0.0022,
        workers=4,
        description="基础大模型，作为性能上限参考",
    ),
    VariantConfig(
        key="rtdetr_mnv4",
        label="RT-DETR-MNV4",
        yaml_file="rtdetr-mnv4-hybrid-m.yaml",
        batch=8,
        lr0=0.0018,
        workers=4,
        description="MobileNetV4混合主干，轻量基线",
    ),
    VariantConfig(
        key="rtdetr_mnv4_sea",
        label="RT-DETR-MNV4-SEA",
        yaml_file="rtdetr-mnv4-hybrid-m-sea.yaml",
        batch=6,
        lr0=0.0015,
        workers=4,
        description="加入SEA注意力的改进版",
    ),
    VariantConfig(
        key="rtdetr_mnv4_sea_bifpn",
        label="RT-DETR-MNV4-SEA-BiFPN",
        yaml_file="rtdetr-mnv4-hybrid-m-sea-bifpn-lite.yaml",
        batch=6,
        lr0=0.0016,
        workers=4,
        description="SEA + BiFPN-Lite 融合版本",
    ),
    VariantConfig(
        key="rtdetr_mnv4_sea_asff",
        label="RT-DETR-MNV4-SEA-ASFF",
        yaml_file="rtdetr-mnv4-hybrid-m-sea-asff-dysample.yaml",
        batch=8,
        lr0=0.0017,
        workers=4,
        description="SEA + ASFF + DySample 轻量高效融合版本（v1-失败）",
    ),
    VariantConfig(
        key="rtdetr_mnv4_sea_asff_v2",
        label="RT-DETR-MNV4-SEA-ASFF-v2",
        yaml_file="rtdetr-mnv4-hybrid-m-sea-asff-v2.yaml",
        batch=8,
        lr0=0.0016,
        workers=4,
        description="SEA + 完整三尺度ASFF，256通道，RepC3×3（v2-完整版）",
    ),
    VariantConfig(
        key="rtdetr_mnv4_sea_asff_v3",
        label="RT-DETR-MNV4-SEA-ASFF-v3",
        yaml_file="rtdetr-mnv4-hybrid-m-sea-asff-v3.yaml",
        batch=8,
        lr0=0.0016,
        workers=4,
        description="SEA + 完整三尺度ASFF，224通道，RepC3×2（v3-轻量版）⭐ 推荐",
    ),
]


def build_train_config(
    variant: VariantConfig,
    args: argparse.Namespace,
    batch_size: int,
) -> Dict[str, Any]:
    """构建 Ultralytics 训练参数字典。"""

    batch_size = max(1, batch_size)

    base = {
        "task": "detect",
        "mode": "train",
        "model": str(MODEL_DIR / variant.yaml_file),
        "data": DATA_CONFIG,
        "epochs": args.epochs,
        "batch": batch_size,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "device": args.device,
        "workers": variant.workers,
        "amp": True,
        "cache": "ram",
        "rect": True,
        "optimizer": "AdamW",
        "lr0": variant.lr0,
        "lrf": 0.0015,
        "momentum": 0.94,
        "weight_decay": 0.00045,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": True,
        "hsv_h": 0.015,
        "hsv_s": 0.65,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "box": 7.5,
        "cls": 0.55,
        "dfl": 1.5,
        "val": True,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 400,
        "save": True,
        "save_period": args.save_period,
        "project": args.project,
        "name": f"{variant.key}_{args.tag}_bs{batch_size}",
        "exist_ok": True,
        "verbose": True,
        "seed": args.seed,
        "deterministic": False,
        "plots": True,
        "close_mosaic": 10,
        "overlap_mask": True,
        "mask_ratio": 4,
        "profile": False,
        "half": False,
        "dnn": False,
    }
    return base


def extract_metrics(results: Any) -> Dict[str, float]:
    """从 Ultralytics 结果对象中提取常用指标。"""

    metrics = {}
    candidates = []

    if results is None:
        return metrics
    if hasattr(results, "metrics") and isinstance(results.metrics, dict):
        candidates.append(results.metrics)
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        candidates.append(results.results_dict)
    if hasattr(results, "__dict__"):
        raw_dict = {k: v for k, v in vars(results).items() if isinstance(v, dict)}
        candidates.extend(raw_dict.values())

    for data in candidates:
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
    return metrics


def pick_metric(metrics: Dict[str, float], keys) -> Optional[float]:
    """按照优先级返回第一个可用指标。"""

    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def run_variant(variant: VariantConfig, args: argparse.Namespace) -> Dict[str, float]:
    """训练单个变体并返回指标。"""

    base_batch = args.batch if args.batch is not None else variant.batch
    attempt_batch = max(1, int(round(base_batch * args.batch_scale)))
    min_batch = max(1, args.min_batch)
    
    # 提前创建权重目录避免保存失败
    weights_root = Path(args.project) / f"{variant.key}_{args.tag}_bs{attempt_batch}"
    (weights_root / "weights").mkdir(parents=True, exist_ok=True)

    while True:
        train_cfg = build_train_config(variant, args, batch_size=attempt_batch)
        print("=" * 80)
        print(f"🎯 训练 {variant.label}")
        print(f"📝 YAML: {train_cfg['model']}")
        print(f"📦 项目: {train_cfg['project']} / {train_cfg['name']}")
        print(f"🧮 Batch Size: {train_cfg['batch']} (min {min_batch})")

        model: Optional[RTDETR] = None
        try:
            model = RTDETR(train_cfg["model"])
            results = model.train(**{k: v for k, v in train_cfg.items() if k != "model"})
            metrics = extract_metrics(results)
            return metrics
        except RuntimeError as exc:
            message = str(exc).lower()
            if "invalid argument" in message:
                print("⚠️ 检测到可能的数据或数值异常，可尝试开启 CUDA_LAUNCH_BLOCKING=1 并检查标签。")
            if "out of memory" in message and attempt_batch > min_batch:
                attempt_batch = max(min_batch, max(1, attempt_batch // 2))
                print(f"⚠️ 显存不足，降至 batch={attempt_batch} 后重试...")
                continue
            raise
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并对比 RT-DETR 四个变体")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入分辨率")
    parser.add_argument("--patience", type=int, default=25, help="早停耐心值")
    parser.add_argument("--device", default="0", help="训练使用的设备标识")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Ultralytics 结果输出目录")
    parser.add_argument("--tag", default="single", help="run 名称后缀")
    parser.add_argument("--save-period", dest="save_period", type=int, default=20, help="检查点保存间隔 (epoch)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--batch",
        type=int,
        help="覆盖默认 batch 大小",
    )
    parser.add_argument(
        "--batch-scale",
        type=float,
        default=1.0,
        help="在默认/自定义 batch 基础上乘以该系数",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=2,
        help="自动退避时允许的最小 batch",
    )
    parser.add_argument(
        "--variant",
        choices=[v.key for v in VARIANTS],
        help="可选覆盖脚本内的 SELECTED_VARIANT_KEY",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    variant_key = args.variant or SELECTED_VARIANT_KEY
    variant = next((v for v in VARIANTS if v.key == variant_key), None)
    if variant is None:
        raise SystemExit(
            f"未找到要训练的模型: {variant_key}，可选: {[v.key for v in VARIANTS]}"
        )

    # torch.autograd.set_detect_anomaly(True)  # 禁用异常检测以避免梯度计算异常中断训练
    
    print(f"\n🧭 当前训练模型: {variant.label} ({variant_key})")
    print(f"📝 配置: batch={args.batch or variant.batch}, lr0={variant.lr0}, workers={variant.workers}")
    print("=" * 80)

    metrics = run_variant(variant, args)
    print(f"\n📊 {variant.label} 指标: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

    if metrics:
        output_dir = Path(args.project) / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"{variant.key}_{args.tag}_{timestamp}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "variant": variant.label,
                    "metrics": metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"📄 训练指标已保存至: {report_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
