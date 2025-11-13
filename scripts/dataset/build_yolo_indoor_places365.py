#!/usr/bin/env python3
"""
Build YOLO-Indoor subset by filtering images via Places365 Indoor/Outdoor.

- Loads a Places365 ResNet-50 classifier and maps predictions to Indoor/Outdoor
- Filters COCO images in YOLO format predicted as Indoor
- Creates new directory structure with filtered images and labels

Usage:
  python scripts/dataset/build_yolo_indoor_places365.py \
    --data-root /root/autodl-tmp/database/coco/coco \
    --output-root datasets/coco_indoor_yolo \
    --batch-size 128 --topk 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# -------------------------
# Constants and Download URLs
# -------------------------
PLACES365_WEIGHTS_URL = (
    "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
)
CATEGORIES_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
)
IO_MAP_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt"
)
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".cache", "places365")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(url: str, dst: Path, desc: str) -> None:
    import urllib.request
    import urllib.error

    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and has non-zero size
    if dst.exists() and dst.stat().st_size > 0:
        print(f"{desc} already exists: {dst}")
        return
    
    # Remove empty/corrupted file if exists
    if dst.exists():
        print(f"Removing empty/corrupted file: {dst}")
        dst.unlink()
    
    print(f"Downloading {desc} from {url} -> {dst}")
    try:
        urllib.request.urlretrieve(url, dst)  # nosec - user-initiated model/data fetch
        
        # Verify download was successful
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(f"Download failed: {dst} is empty or missing")
        
        print(f"Successfully downloaded {desc} ({dst.stat().st_size} bytes)")
        
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as e:
        if dst.exists():
            dst.unlink()  # Clean up corrupted file
        raise RuntimeError(f"Failed to download {desc} from {url}: {e}")


def load_places365_categories(categories_path: Path) -> List[str]:
    cats = []
    with categories_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) == 1:
                cats.append(parts[0])
            else:
                cats.append(parts[-1].strip())
    # Normalize leading '/' variants
    cats = [c.split("/")[-1] if "/" in c else c for c in cats]
    return cats


def load_places365_io_map(io_path: Path, num_classes: int) -> List[int]:
    io = []
    with io_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) >= 2:
                try:
                    # IO值：1=室内, 2=室外，转换为 1=室内, 0=室外
                    io_value = int(parts[1])
                    io.append(1 if io_value == 1 else 0)
                except ValueError:
                    io.append(0)  # 默认室外
    
    if len(io) < num_classes:
        io += [0] * (num_classes - len(io))  # 默认室外
    return io[:num_classes]


def build_places365_resnet50(weights_path: Path, device: torch.device) -> nn.Module:
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 365)

    # Verify weights file exists and has reasonable size
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    file_size = weights_path.stat().st_size
    if file_size < 1024 * 1024:  # Less than 1MB is likely corrupted
        raise RuntimeError(f"Weights file appears corrupted (size: {file_size} bytes): {weights_path}")
    
    print(f"Loading Places365 weights from {weights_path} (size: {file_size} bytes)")
    
    try:
        ckpt = torch.load(weights_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")
    
    state = ckpt.get("state_dict", ckpt)

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] Missing keys when loading Places365 weights: {missing[:5]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys when loading Places365 weights: {unexpected[:5]} ...")

    model.eval().to(device)
    return model


def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def infer_indoor_flags(
    image_paths: List[Path],
    model: nn.Module,
    categories: List[str],
    io_map: List[int],
    device: torch.device,
    batch_size: int = 128,
    topk: int = 5,
) -> List[bool]:
    tfm = build_transform()
    indoor_flags: List[bool] = []

    def decide_indoor(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        topk_vals, topk_idx = probs.topk(k=min(topk, probs.size(1)), dim=1)
        io_tensor = torch.tensor([io_map[i] for i in range(len(io_map))], device=probs.device)
        topk_io = io_tensor[topk_idx]
        
                # 使用平衡的室内判断标准（扩充数据集版本）：
        # 1. 加权 IO 分数 >= 0.6 (大幅降低，原来是0.9)
        # 2. 最高概率的类别必须是室内 或者 前2的平均IO > 0.7
        # 3. 最高概率必须 >= 0.25 (降低阈值，原来是0.4)
        # 4. 移除Top-3全部室内的限制
        # 5. 只排除明显的室外类别
        
        weighted_io = (topk_io * topk_vals).sum(dim=1) / topk_vals.sum(dim=1).clamp(min=1e-6)
        top1_indoor = io_tensor[topk_idx[:, 0]] == 1  # 最高概率是室内类别
        top2_avg_io = topk_io[:, :2].mean(dim=1) > 0.7  # Top-2平均室内概率高
        top1_conf = topk_vals[:, 0] >= 0.25  # 降低置信度要求
        
        # 只排除明显的室外类别
        outdoor_categories = [
            'street', 'road', 'highway', 'bridge', 'parking_lot',
            'gas_station', 'beach', 'ocean', 'lake', 'river',
            'mountain', 'forest', 'field', 'sky', 'playground'
        ]
        
        # 检查top1是否为明显室外类别
        top1_not_outdoor = True
        for batch_idx in range(topk_idx.size(0)):
            top1_cat_idx = topk_idx[batch_idx, 0].item()
            top1_category = categories[top1_cat_idx].lower()
            for outdoor_cat in outdoor_categories:
                if outdoor_cat in top1_category:
                    top1_not_outdoor = False
                    break
        
        top1_not_outdoor_tensor = torch.tensor(top1_not_outdoor, device=device)
        
        balanced_indoor = (
            (weighted_io >= 0.6) & 
            (top1_indoor | top2_avg_io) & 
            top1_conf & 
            top1_not_outdoor_tensor
        )
        return balanced_indoor
        
        weighted_io = (topk_io * topk_vals).sum(dim=1) / topk_vals.sum(dim=1).clamp(min=1e-6)
        top1_indoor = io_tensor[topk_idx[:, 0]] == 1  # 最高概率是室内类别
        top1_conf = topk_vals[:, 0] >= 0.4  # 最高概率足够高
        top3_all_indoor = topk_io[:, :3].sum(dim=1) == 3  # Top-3全部室内
        
        # 排除可能有问题的室内类别（通过类别名称检查）
        # 这些类别虽然标记为室内，但可能包含室外元素
        problematic_categories = [
            'balcony', 'porch', 'terrace', 'veranda', 'courtyard',
            'garage', 'parking', 'station', 'platform', 'terminal',
            'hangar', 'warehouse', 'factory', 'construction'
        ]
        
        # 检查top1是否为问题类别
        top1_safe = True
        for batch_idx in range(topk_idx.size(0)):
            top1_cat_idx = topk_idx[batch_idx, 0].item()
            top1_category = categories[top1_cat_idx].lower()
            for prob_cat in problematic_categories:
                if prob_cat in top1_category:
                    top1_safe = False
                    break
        
        top1_safe_tensor = torch.tensor(top1_safe, device=device)
        
        ultra_strict_indoor = (
            (weighted_io >= 0.9) & 
            top1_indoor & 
            top1_conf & 
            top3_all_indoor & 
            top1_safe_tensor
        )
        return ultra_strict_indoor

    batch: List[torch.Tensor] = []
    paths_batch: List[Path] = []
    
    for p in tqdm(image_paths, desc="Places365 filtering", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            batch.append(tfm(img))
            paths_batch.append(p)
        except Exception:
            indoor_flags.append(False)
            continue

        if len(batch) == batch_size:
            inp = torch.stack(batch, dim=0).to(device, non_blocking=True)
            logits = model(inp)
            flags = decide_indoor(logits).tolist()
            indoor_flags.extend(flags)
            batch.clear()
            paths_batch.clear()

    if batch:
        inp = torch.stack(batch, dim=0).to(device, non_blocking=True)
        logits = model(inp)
        flags = decide_indoor(logits).tolist()
        indoor_flags.extend(flags)

    return indoor_flags[: len(image_paths)]


def process_yolo_split(
    images_dir: Path, 
    labels_dir: Path, 
    output_dir: Path, 
    model: nn.Module,
    categories: List[str],
    io_map: List[int],
    device: torch.device,
    batch_size: int,
    topk: int,
    split_name: str,
    limit: int = 0
):
    """Process a YOLO format split (train2017 or val2017)"""
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(images_dir.glob(ext))
    
    if limit > 0:
        image_files = image_files[:limit]
    
    print(f"\n[{split_name}] Found {len(image_files)} images")
    
    # Filter indoor images with detailed progress
    print(f"[{split_name}] Running Places365 indoor classification...")
    indoor_flags = infer_indoor_flags(
        image_files, model, categories, io_map, device, batch_size, topk
    )
    
    # Calculate statistics
    indoor_count = sum(indoor_flags)
    outdoor_count = len(indoor_flags) - indoor_count
    indoor_rate = indoor_count / len(indoor_flags) * 100 if len(indoor_flags) > 0 else 0
    
    print(f"[{split_name}] Classification results:")
    print(f"  📍 Indoor:  {indoor_count:4d} images ({indoor_rate:.1f}%)")
    print(f"  🌳 Outdoor: {outdoor_count:4d} images ({100-indoor_rate:.1f}%)")
    
    # Create output directories
    out_images_dir = output_dir / "images" / split_name
    out_labels_dir = output_dir / "labels" / split_name
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy indoor images and labels with progress bar
    print(f"[{split_name}] Copying {indoor_count} indoor images and labels...")
    copied_count = 0
    
    with tqdm(total=indoor_count, desc=f"Copying {split_name} indoor files", unit="files") as pbar:
        for img_path, is_indoor in zip(image_files, indoor_flags):
            if is_indoor:
                # Copy image
                dst_img = out_images_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Copy corresponding label file
                label_name = img_path.stem + ".txt"
                src_label = labels_dir / label_name
                if src_label.exists():
                    dst_label = out_labels_dir / label_name
                    shutil.copy2(src_label, dst_label)
                
                copied_count += 1
                pbar.update(1)
    
    print(f"[{split_name}] ✅ Successfully copied {copied_count} indoor image pairs")
    return copied_count


def create_yaml_config(output_dir: Path, num_classes: int = 80):
    """Create YOLO dataset configuration file"""
    yaml_content = f"""# COCO Indoor Dataset Configuration
# Generated by build_yolo_indoor_places365.py

path: {output_dir.absolute()}  # dataset root dir
train: images/train2017  # train images (relative to 'path')
val: images/val2017      # val images (relative to 'path')

# Classes (COCO format)
nc: {num_classes}  # number of classes
names:
"""
    
    # Add COCO class names
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    for i, name in enumerate(coco_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = output_dir / "coco_indoor.yaml"
    with yaml_path.open("w") as f:
        f.write(yaml_content)
    
    print(f"Created YOLO config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Build YOLO-Indoor via Places365 filtering")
    parser.add_argument("--data-root", type=str, required=True, help="Root path to COCO YOLO format data")
    parser.add_argument("--output-root", type=str, required=True, help="Output directory for filtered dataset")
    
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache dir for Places365 assets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5, help="Top-K categories for IO vote")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Debug: only process first N images per split")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    # Setup cache and download assets
    cache_dir = ensure_dir(args.cache_dir)
    weights_path = cache_dir / "resnet50_places365.pth.tar"
    categories_path = cache_dir / "categories_places365.txt"
    io_path = cache_dir / "IO_places365.txt"

    try:
        if not weights_path.exists():
            download_file(PLACES365_WEIGHTS_URL, weights_path, "Places365 ResNet-50 weights")
        if not categories_path.exists():
            download_file(CATEGORIES_URL, categories_path, "Places365 categories")
        if not io_path.exists():
            download_file(IO_MAP_URL, io_path, "Places365 IO map")
    except Exception as e:
        print(f"[warn] Failed to download Places365 assets: {e}")
        return

    # Load model and assets
    categories = load_places365_categories(categories_path)
    io_map = load_places365_io_map(io_path, num_classes=365)
    device = torch.device(args.device)
    model = build_places365_resnet50(weights_path, device)

    print(f"Using device: {device}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"=" * 60)

    t0 = time.time()
    total_indoor = 0

    # 计算合理的处理数量以保持训练/验证比例
    # COCO原始比例: train2017约118K, val2017约5K (约95:5)
    # 我们调整为更合理的8:2比例
    if args.limit > 0:
        train_limit = int(args.limit * 0.8)  # 80%用于训练
        val_limit = int(args.limit * 0.2)    # 20%用于验证
    else:
        # 处理更多数据以获得足够的室内图像
        # 目标：获得约4000张高质量训练图像 + 1000张验证图像
        # 预期过滤率约30-40%（更严格的过滤），所以需要处理约12000-15000张原始图像
        train_limit = 15000   # 预期获得约3000-4500张训练图像
        val_limit = 5000      # 预期获得约1000-2000张验证图像
    
    print(f"Target processing: {train_limit} train + {val_limit} val = {train_limit + val_limit} total images")

    # Process train2017
    train_imgs = data_root / "images" / "train2017"
    train_labels = data_root / "labels" / "train2017"
    if train_imgs.exists() and train_labels.exists():
        train_count = process_yolo_split(
            train_imgs, train_labels, output_root, model, categories, io_map,
            device, args.batch_size, args.topk, "train2017", train_limit
        )
        total_indoor += train_count
    else:
        print("[warn] train2017 not found")

    # Process val2017
    val_imgs = data_root / "images" / "val2017"
    val_labels = data_root / "labels" / "val2017"
    if val_imgs.exists() and val_labels.exists():
        val_count = process_yolo_split(
            val_imgs, val_labels, output_root, model, categories, io_map,
            device, args.batch_size, args.topk, "val2017", val_limit
        )
        total_indoor += val_count
    else:
        print("[warn] val2017 not found")

    # Create YAML config
    create_yaml_config(output_root)

    elapsed = (time.time() - t0) / 60.0
    print(f"\n" + "=" * 60)
    print(f"🎉 Processing completed in {elapsed:.1f} minutes")
    print(f"📊 Total indoor images: {total_indoor}")
    
    # 显示最终统计
    final_train = len(list((output_root / "images" / "train2017").glob("*.jpg")))
    final_val = len(list((output_root / "images" / "val2017").glob("*.jpg")))
    if final_train + final_val > 0:
        train_ratio = final_train / (final_train + final_val) * 100
        val_ratio = final_val / (final_train + final_val) * 100
        print(f"📁 Final dataset composition:")
        print(f"   Training:   {final_train:4d} images ({train_ratio:.1f}%)")
        print(f"   Validation: {final_val:4d} images ({val_ratio:.1f}%)")
    
    print(f"💾 Indoor dataset saved to: {output_root}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()