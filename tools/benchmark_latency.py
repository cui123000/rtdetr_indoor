#!/usr/bin/env python3
"""
Benchmark latency and CUDA max memory for Ultralytics models.
Usage:
  python tools/benchmark_latency.py --weights path/to/best.pt --imgsz 640 --iters 200 --warmup 20 --device 0
Multiple weights can be provided comma-separated.
Outputs CSV columns:
  model, source, imgsz, iters, warmup, device, median_ms, p95_ms, throughput_fps, max_mem_gb
"""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import time
from typing import List
from pathlib import Path

import torch, sys
# Prefer local ultralytics repo (project copy) so custom MobileNetV4 modules are visible
try:
    REPO_ROOT = Path(__file__).resolve().parents[1] / 'ultralytics'
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
        print(f"[INFO] Added local ultralytics path: {REPO_ROOT}")
    if 'ultralytics' in sys.modules:
        del sys.modules['ultralytics']
        print("[INFO] Cleared preloaded 'ultralytics' for reload")
except Exception:
    pass
def _ensure_cv2_stub():
    try:
        import cv2  # noqa: F401
        return
    except Exception as exc:
        import types
        stub = types.ModuleType("cv2")
        def _noop(*args, **kwargs): return None
        for a in ["imread","imwrite","imshow","resize","cvtColor","waitKey","destroyAllWindows"]:
            setattr(stub, a, _noop)
        stub.COLOR_BGR2RGB = 0
        stub.INTER_LINEAR = 1
        sys.modules['cv2'] = stub
        print(f"[WARN] OpenCV unavailable ({exc}); using stub for latency benchmarking.")
_ensure_cv2_stub()
try:
    from ultralytics import YOLO, RTDETR
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
    print("[ERROR] ultralytics not found. Please 'pip install ultralytics'.")
    raise e


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def run_once(model, imgsz, device):
    with torch.no_grad():
        x = torch.randn(1, 3, imgsz, imgsz, device=device)
        y = model(x) if hasattr(model, '__call__') else model.model(x)
        return y


def measure(weights: str, imgsz: int, iters: int, warmup: int, device: str, out: str):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    paths = [w.strip() for w in weights.split(',') if w.strip()]
    rows = []

    for wp in paths:
        print(f"[INFO] Loading: {wp}")
        if 'rtdetr' in wp.lower():
            model = RTDETR(wp)
        else:
            model = YOLO(wp)
        torch_model = model.model if hasattr(model, 'model') else model
        torch_model.to(device)
        # warmup
        with torch.no_grad():
            for _ in range(max(1, warmup)):
                x = torch.randn(1, 3, imgsz, imgsz, device=device)
                _ = torch_model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        # measure
        times = []
        mem_peak = 0
        with torch.no_grad():
            for _ in range(iters):
                x = torch.randn(1, 3, imgsz, imgsz, device=device)
                t0 = time_sync()
                _ = torch_model(x)
                t1 = time_sync()
                times.append((t1 - t0) * 1000.0)
                if torch.cuda.is_available():
                    mem_peak = max(mem_peak, torch.cuda.max_memory_allocated() / (1024 ** 3))
        median_ms = round(statistics.median(times), 3)
        p95_ms = round(statistics.quantiles(times, n=20)[-1], 3) if len(times) >= 20 else round(sorted(times)[int(0.95 * (len(times)-1))], 3)
        throughput = round(1000.0 / median_ms, 2) if median_ms > 0 else 0.0
        rows.append({
            'model': os.path.basename(wp),
            'source': wp,
            'imgsz': imgsz,
            'iters': iters,
            'warmup': warmup,
            'device': device,
            'median_ms': median_ms,
            'p95_ms': p95_ms,
            'throughput_fps': throughput,
            'max_mem_gb': round(mem_peak, 3)
        })
    # write CSV
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Saved: {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='Comma separated list of weights or YAML')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=str, default='tools/latency_results.csv')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    measure(args.weights, args.imgsz, args.iters, args.warmup, args.device, args.out)
