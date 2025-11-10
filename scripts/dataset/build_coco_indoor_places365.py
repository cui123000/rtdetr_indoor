#!/usr/bin/env python3
"""
Build COCO-Indoor subset by filtering images via Places365 Indoor/Outdoor.

- Loads a Places365 ResNet-50 classifier (365 classes) and maps top-k scene predictions
  to Indoor/Outdoor using the official IO mapping list.
- Filters COCO images predicted as Indoor and prunes annotations accordingly.

Usage (example):
  python scripts/dataset/build_coco_indoor_places365.py \
    --train-images /path/to/coco/train2017 \
    --train-anno   /path/to/coco/annotations/instances_train2017.json \
    --val-images   /path/to/coco/val2017 \
    --val-anno     /path/to/coco/annotations/instances_val2017.json \
    --out-train    datasets/coco_indoor/annotations/instances_train2017_coco_indoor.json \
    --out-val      datasets/coco_indoor/annotations/instances_val2017_coco_indoor.json \
    --batch-size 128 --topk 5

Notes:
- Will attempt to download Places365 weights and category/IO mapping if not present.
- Requires: torch, torchvision, pillow, tqdm, pycocotools (for COCO JSON structure convenience).
- For speed, set CUDA_VISIBLE_DEVICES and use a GPU if available.
"""
from __future__ import annotations

import argparse
import json
import os
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

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"Downloading {desc} from {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)  # nosec - user-initiated model/data fetch


def load_places365_categories(categories_path: Path) -> List[str]:
    cats = []
    with categories_path.open("r", encoding="utf-8") as f:
        for line in f:
            # line format: '0 abbey' or '0 /a/abbey' depending on file variant
            line = line.strip()
            if not line:
                continue
            # official file uses 'index category_name' or 'category_name' per repo variant
            parts = line.split(" ")
            if len(parts) == 1:
                cats.append(parts[0])
            else:
                # Some versions: '0 /a/abbey' -> keep the name part
                cats.append(parts[-1].strip())
    # Normalize leading '/' variants (e.g., '/a/abbey' -> 'abbey')
    cats = [c.split("/")[-1] if "/" in c else c for c in cats]
    return cats


def load_places365_io_map(io_path: Path, num_classes: int) -> List[int]:
    io = []
    with io_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line is '0' or '1' per class index order (0=indoor, 1=outdoor) per official file
            try:
                io.append(int(line))
            except ValueError:
                # Some variants may include '2' (ambiguous). Treat 2 as outdoor-leaning (1) to be conservative.
                try:
                    v = int(float(line))
                    io.append(1 if v != 0 else 0)
                except Exception:
                    io.append(1)
    if len(io) < num_classes:
        # pad conservatively as outdoor
        io += [1] * (num_classes - len(io))
    return io[:num_classes]


def build_places365_resnet50(weights_path: Path, device: torch.device) -> nn.Module:
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    # Replace final fc to 365 classes
    model.fc = nn.Linear(model.fc.in_features, 365)

    # Load checkpoint
    ckpt = torch.load(weights_path, map_location="cpu")
    # Some checkpoints use 'state_dict' key
    state = ckpt.get("state_dict", ckpt)

    # Strip 'module.' prefix if present
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
        # logits: [B, 365]
        probs = torch.softmax(logits, dim=1)
        topk_vals, topk_idx = probs.topk(k=min(topk, probs.size(1)), dim=1)
        # Map indices to IO labels (0 indoor, 1 outdoor). We'll compute a weighted score.
        io_tensor = torch.tensor([io_map[i] for i in range(len(io_map))], device=probs.device)
        # [B, K] gather IO labels
        topk_io = io_tensor[topk_idx]
        # Weighted decision: if weighted avg IO < 0.5 -> indoor
        weighted_io = (topk_io * topk_vals).sum(dim=1) / topk_vals.sum(dim=1).clamp(min=1e-6)
        return (weighted_io < 0.5)

    # Mini-batch processing
    batch: List[torch.Tensor] = []
    paths_batch: List[Path] = []
    for p in tqdm(image_paths, desc="Places365 filtering", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            batch.append(tfm(img))
            paths_batch.append(p)
        except Exception:
            # If image cannot be read, mark as outdoor (skip)
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

    # Truncate in case of any mismatch due to unreadable images logic
    return indoor_flags[: len(image_paths)]


def load_coco_json(coco_json_path: Path) -> Dict:
    with coco_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prune_coco(coco: Dict, keep_image_ids: set[int]) -> Dict:
    images = [img for img in coco.get("images", []) if img.get("id") in keep_image_ids]
    annos = [ann for ann in coco.get("annotations", []) if ann.get("image_id") in keep_image_ids]
    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = images
    out["annotations"] = annos
    return out


def gather_image_paths(coco: Dict, images_root: Path) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    image_ids: List[int] = []
    for img in coco.get("images", []):
        fn = img.get("file_name")
        img_id = img.get("id")
        if fn is None or img_id is None:
            continue
        p = images_root / fn
        paths.append(p)
        image_ids.append(int(img_id))
    return paths, image_ids


def main():
    parser = argparse.ArgumentParser(description="Build COCO-Indoor via Places365 IO filtering")
    parser.add_argument("--train-images", type=str, default=None, help="Path to COCO train2017 images dir")
    parser.add_argument("--train-anno", type=str, default=None, help="Path to COCO instances_train2017.json")
    parser.add_argument("--val-images", type=str, default=None, help="Path to COCO val2017 images dir")
    parser.add_argument("--val-anno", type=str, default=None, help="Path to COCO instances_val2017.json")
    parser.add_argument("--out-train", type=str, default=None, help="Output path for filtered train JSON")
    parser.add_argument("--out-val", type=str, default=None, help="Output path for filtered val JSON")

    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Cache dir for Places365 assets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5, help="Top-K categories for IO vote")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Debug: only process first N images")

    args = parser.parse_args()

    cache_dir = ensure_dir(args.cache_dir)
    weights_path = cache_dir / "resnet50_places365.pth.tar"
    categories_path = cache_dir / "categories_places365.txt"
    io_path = cache_dir / "IO_places365.txt"

    # Download assets if missing
    try:
        if not weights_path.exists():
            download_file(PLACES365_WEIGHTS_URL, weights_path, "Places365 ResNet-50 weights")
        if not categories_path.exists():
            download_file(CATEGORIES_URL, categories_path, "Places365 categories")
        if not io_path.exists():
            download_file(IO_MAP_URL, io_path, "Places365 IO map")
    except Exception as e:
        print(f"[warn] Failed to download one or more Places365 assets: {e}")
        print("You can manually place files into:", cache_dir)
        print("  - resnet50_places365.pth.tar\n  - categories_places365.txt\n  - IO_places365.txt")

    # Load assets
    categories = load_places365_categories(categories_path)
    if len(categories) != 365:
        print(f"[warn] categories loaded: {len(categories)} (expected 365)")
    io_map = load_places365_io_map(io_path, num_classes=365)

    device = torch.device(args.device)
    model = build_places365_resnet50(weights_path, device)

    def process_split(images_dir: str | None, anno_path: str | None, out_json: str | None, split_name: str):
        if not images_dir or not anno_path or not out_json:
            print(f"[skip] {split_name}: paths not provided")
            return
        images_root = Path(images_dir)
        coco = load_coco_json(Path(anno_path))
        paths, image_ids = gather_image_paths(coco, images_root)
        if args.limit and args.limit > 0:
            paths = paths[: args.limit]
            image_ids = image_ids[: args.limit]
        print(f"[{split_name}] Total images to evaluate: {len(paths)}")
        flags = infer_indoor_flags(paths, model, categories, io_map, device, args.batch_size, args.topk)
        keep_ids = {img_id for img_id, is_indoor in zip(image_ids, flags) if is_indoor}
        print(f"[{split_name}] Indoor images kept: {len(keep_ids)} / {len(image_ids)} ({100.0*len(keep_ids)/max(1,len(image_ids)):.2f}%)")
        pruned = prune_coco(coco, keep_ids)
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(pruned, f)
        print(f"[{split_name}] Saved filtered COCO JSON -> {out_path}")

    t0 = time.time()
    process_split(args.train_images, args.train_anno, args.out_train, "train")
    process_split(args.val_images, args.val_anno, args.out_val, "val")
    print(f"Done in {(time.time()-t0)/60.0:.1f} min")


if __name__ == "__main__":
    main()
