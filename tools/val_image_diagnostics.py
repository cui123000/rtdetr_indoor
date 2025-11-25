import sys
from pathlib import Path
import os
import yaml
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / 'ultralytics') not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from ultralytics import RTDETR
except Exception:
    # Attempt to pre-register and load the repository's inner ultralytics package
    try:
        import importlib.util
        local_init = repo_root / 'ultralytics' / 'ultralytics' / '__init__.py'
        if local_init.exists():
            spec = importlib.util.spec_from_file_location('local_ultralytics', str(local_init))
            local_mod = importlib.util.module_from_spec(spec)
            local_mod.__path__ = [str(repo_root / 'ultralytics' / 'ultralytics')]
            import sys as _sys
            _sys.modules['ultralytics'] = local_mod
            spec.loader.exec_module(local_mod)
            from ultralytics import RTDETR
        else:
            from ultralytics.models.rtdetr import RTDETR
    except Exception:
        from ultralytics.models.rtdetr import RTDETR


def load_dataset_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_labels_for_image(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append((cls, x, y, w, h))
    return boxes


def xywh_to_xyxy_rel(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    # a: [x1,y1,x2,y2], b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_predictions(preds):
    # Return list of tuples (x1,y1,x2,y2,conf,cls)
    res = None
    if hasattr(preds, 'boxes'):
        res = preds
    elif isinstance(preds, (list, tuple)) and len(preds) > 0:
        res = preds[0]
    out = []
    if res is None:
        return out
    boxes = getattr(res, 'boxes', None)
    # Robust handling: convert Boxes to CPU numpy arrays then iterate
    if boxes is not None:
        try:
            # Boxes often expose .xyxy, .conf, .cls as tensors; prefer numpy
            xyxy_all = None
            conf_all = None
            cls_all = None
            if hasattr(boxes, 'xyxy'):
                xyxy_all = boxes.xyxy.cpu().numpy()
            if hasattr(boxes, 'conf'):
                conf_all = boxes.conf.cpu().numpy()
            if hasattr(boxes, 'cls'):
                cls_all = boxes.cls.cpu().numpy()
            # if we obtained arrays, iterate by index
            if xyxy_all is not None:
                n = xyxy_all.shape[0]
                for i in range(n):
                    xy = xyxy_all[i].tolist()
                    conf = float(conf_all[i]) if conf_all is not None else 0.0
                    cls = int(cls_all[i]) if cls_all is not None else -1
                    out.append((xy[0], xy[1], xy[2], xy[3], conf, cls))
                return out
        except Exception:
            # fallback to old iteration style
            pass
        for b in boxes:
            try:
                xyxy = b.xyxy.tolist() if hasattr(b, 'xyxy') else list(b[:4])
                conf = float(b.conf) if hasattr(b, 'conf') else (float(b[4]) if len(b) > 4 else 0.0)
                cls = int(b.cls) if hasattr(b, 'cls') else (int(b[5]) if len(b) > 5 else -1)
                out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls))
            except Exception:
                continue
    else:
        try:
            if isinstance(res, (list, tuple)) and len(res) >= 1:
                boxes_t = res[0]
                scores_t = res[1] if len(res) > 1 else None
                classes_t = res[2] if len(res) > 2 else None
                n = int(getattr(boxes_t, 'shape', [0])[0])
                for i in range(n):
                    xyxy = [float(x) for x in boxes_t[i].tolist()]
                    conf = float(scores_t[i]) if scores_t is not None else 0.0
                    cls = int(classes_t[i]) if classes_t is not None else -1
                    out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls))
        except Exception:
            return out
    return out


def run(weights, data_yaml, out_dir, topk=20, conf_thres=0.001, imgsz=640):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset_yaml(data_yaml)
    val_dir = Path(data['path']) / data['val']
    label_dir = Path(data['path']) / data.get('labels', data.get('labels_dir', 'labels'))

    model = RTDETR(weights)

    records = []
    all_conf = []
    all_best_iou = []

    img_files = sorted(list(val_dir.glob('*.jpg')))
    for img_path in img_files:
        img = str(img_path)
        # load gt
        img_pil = None
        try:
            from PIL import Image
            img_pil = Image.open(img_path)
        except Exception:
            pass
        img_w, img_h = (img_pil.size if img_pil is not None else (imgsz, imgsz))
        label_path = (Path(str(img_path).replace(str(val_dir), str(label_dir)))).with_suffix('.txt')
        gt_boxes = load_labels_for_image(label_path)
        gt_xyxy = [xywh_to_xyxy_rel(x, y, w, h, img_w, img_h) for (_, x, y, w, h) in gt_boxes]

        preds = model.predict(source=img, conf=conf_thres, imgsz=imgsz)
        parsed = parse_predictions(preds)
        # sort by conf desc
        parsed_sorted = sorted(parsed, key=lambda x: x[4], reverse=True)

        row = {'image': str(img_path), 'num_gt': len(gt_xyxy), 'num_preds': len(parsed_sorted)}

        best_iou_topk = 0.0
        for k in range(topk):
            key_conf = f'pred_{k+1}_conf'
            key_iou = f'pred_{k+1}_best_iou'
            if k < len(parsed_sorted):
                pred = parsed_sorted[k]
                conf = pred[4]
                # compute best iou with any gt
                best_i = 0.0
                for g in gt_xyxy:
                    best_i = max(best_i, iou_xyxy(pred[:4], g))
                row[key_conf] = conf
                row[key_iou] = best_i
                all_conf.append(conf)
                all_best_iou.append(best_i)
                best_iou_topk = max(best_iou_topk, best_i)
            else:
                row[key_conf] = float('nan')
                row[key_iou] = float('nan')

        row['best_iou_topk'] = best_iou_topk
        records.append(row)

    # save csv
    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / 'diagnostics.csv'
    df.to_csv(csv_path, index=False)

    # plots
    plt.figure(figsize=(6, 4))
    plt.hist([c for c in all_conf if not math.isnan(c)], bins=50)
    plt.title('Top-K predictions confidence distribution')
    plt.xlabel('confidence')
    plt.ylabel('count')
    plt.savefig(out_dir / 'conf_hist.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist([i for i in all_best_iou if not math.isnan(i)], bins=50)
    plt.title('Top-K predictions best IoU with GT distribution')
    plt.xlabel('best IoU')
    plt.ylabel('count')
    plt.savefig(out_dir / 'best_iou_hist.png')
    plt.close()

    # scatter conf vs iou
    plt.figure(figsize=(6, 6))
    plt.scatter(all_conf, all_best_iou, s=4, alpha=0.6)
    plt.xlabel('confidence')
    plt.ylabel('best IoU')
    plt.title('confidence vs best IoU (top-K preds)')
    plt.savefig(out_dir / 'conf_vs_iou.png')
    plt.close()

    print('Saved diagnostics csv and plots to', out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default=str(repo_root / 'datasets' / 'homeobjects-3K' / 'HomeObjects-3K.yaml'))
    parser.add_argument('--out', default='runs/detect/visual_ert')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    run(args.weights, args.data, args.out, topk=args.topk, conf_thres=args.conf, imgsz=args.imgsz)
