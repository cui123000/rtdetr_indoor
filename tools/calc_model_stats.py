#!/usr/bin/env python3
"""
Compute parameter counts and FLOPs/MACs for one or more Ultralytics YOLO / RT-DETR YAML or weight files.
Usage:
  python tools/calc_model_stats.py --models path/to/model.yaml,path/to/other.yaml --imgsz 640 --out tools/model_stats.csv

Outputs CSV columns:
  model, source, params_M, flops_G, macs_G, imgsz, device
If FLOPs/MACs computation fails (thop not installed or unsupported ops), flops_G/macs_G will be NA.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from typing import List
from pathlib import Path

# Prefer local ultralytics repo (project copy) so custom MobileNetV4 modules are visible
try:
    REPO_ROOT = Path(__file__).resolve().parents[1] / 'ultralytics'
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
        print(f"[INFO] Added local ultralytics path: {REPO_ROOT}")
    if 'ultralytics' in sys.modules:
        # remove preloaded site-packages version to reload project version
        del sys.modules['ultralytics']
        print("[INFO] Cleared preloaded 'ultralytics' for reload")
except Exception:
    pass

"""CPU-only friendly import: if OpenCV libGL is missing, create a minimal stub so ultralytics can import."""
def _ensure_cv2_stub():
    try:
        import cv2  # noqa: F401
        return
    except Exception as exc:  # missing libGL etc.
        import types
        stub = types.ModuleType("cv2")
        def _noop(*args, **kwargs): return None
        # common symbols used in ultralytics utils
        attrs = ["imread","imwrite","imshow","resize","cvtColor","waitKey","destroyAllWindows"]
        for a in attrs:
            setattr(stub, a, _noop)
        # minimal constants
        stub.COLOR_BGR2RGB = 0
        stub.INTER_LINEAR = 1
        sys.modules['cv2'] = stub
        print(f"[WARN] OpenCV unavailable ({exc}); using stub module. Results (visual ops) disabled.")

_ensure_cv2_stub()
try:
    from ultralytics import YOLO, RTDETR
    # Register custom MobileNetV4 blocks into ultralytics.nn.tasks globals so YAML parse_model can find them
    try:
        import ultralytics.nn.tasks as _ut_tasks
        from ultralytics.nn.modules.mobilenetv4 import (
            EdgeResidual,
            UniversalInvertedResidual,
            Sea_Attention_Simplified,
            OptimizedSEA_Attention,
            TransformerEnhancedSEA,
        )
        for _name, _cls in [
            ("EdgeResidual", EdgeResidual),
            ("UniversalInvertedResidual", UniversalInvertedResidual),
            ("Sea_Attention_Simplified", Sea_Attention_Simplified),
            ("OptimizedSEA_Attention", OptimizedSEA_Attention),
            ("TransformerEnhancedSEA", TransformerEnhancedSEA),
        ]:
            setattr(_ut_tasks, _name, _cls)
    except Exception as _e:
        print(f"[WARN] MobilenetV4 custom modules registration failed or not needed: {_e}")
except ImportError as e:
    print("[ERROR] ultralytics not importable even after cv2 stub. Please 'pip install ultralytics'.")
    raise e

# Optional FLOPs via thop
try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False

import torch

def load_model(path: str, device: str):
    p = str(path).lower()
    if 'rtdetr' in p:
        model = RTDETR(path)
    else:
        model = YOLO(path, task='detect')
    # Force building underlying torch model by a dry run
    imgsz = 64
    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        try:
            model.model(dummy)  # some versions
        except Exception:
            try:
                model(dummy)
            except Exception:
                pass
    return model


def compute_stats(model: YOLO, imgsz: int, device: str):
    torch_model = model.model if hasattr(model, 'model') else model
    params = sum(p.numel() for p in torch_model.parameters())
    params_M = params / 1e6
    flops_G = 'NA'
    macs_G = 'NA'
    if THOP_AVAILABLE:
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        try:
            macs, params_tmp = profile(torch_model, inputs=(dummy,), verbose=False)
            # thop reports macs (multiply-adds). FLOPs approx = 2 * MACs for conv layers; we'll record macs separately.
            macs_G = round(macs / 1e9, 4)
            flops_G = round((macs * 2) / 1e9, 4)
        except Exception as e:
            print(f"[WARN] FLOPs/MACs profiling failed: {e}")
    return params_M, flops_G, macs_G


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', type=str, required=True, help='Comma separated list of YAML or weight files')
    ap.add_argument('--imgsz', type=int, default=640, help='Image size for FLOPs computation')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=str, default='tools/model_stats.csv')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_paths = [m.strip() for m in args.models.split(',') if m.strip()]
    rows = []
    for mp in model_paths:
        print(f"[INFO] Loading model: {mp}")
        t0 = time.time()
        model = load_model(mp, args.device)
        load_t = time.time() - t0
        params_M, flops_G, macs_G = compute_stats(model, args.imgsz, args.device)
        rows.append({
            'model': os.path.basename(mp),
            'source': mp,
            'params_M': round(params_M, 4) if isinstance(params_M, float) else params_M,
            'flops_G': flops_G,
            'macs_G': macs_G,
            'imgsz': args.imgsz,
            'device': args.device,
            'load_time_s': round(load_t, 3)
        })
    # Write CSV
    write_header = not (os.path.isfile(args.out))
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Stats written: {args.out}")

if __name__ == '__main__':
    main()
