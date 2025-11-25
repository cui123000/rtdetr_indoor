import sys
import os
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont

# Ensure local ultralytics package import works
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / 'ultralytics') not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from ultralytics import RTDETR
except Exception:
    # Attempt to load the local package __init__.py directly and register it as 'ultralytics'
    try:
        import importlib.util
        local_init = repo_root / 'ultralytics' / 'ultralytics' / '__init__.py'
        if local_init.exists():
            spec = importlib.util.spec_from_file_location('local_ultralytics', str(local_init))
            local_mod = importlib.util.module_from_spec(spec)
            # set package path and register BEFORE executing to avoid circular imports
            local_mod.__path__ = [str(repo_root / 'ultralytics' / 'ultralytics')]
            import sys
            sys.modules['ultralytics'] = local_mod
            # execute the local __init__.py to populate the module
            spec.loader.exec_module(local_mod)
            from ultralytics import RTDETR
        else:
            # fallback to importing directly from models submodule
            from ultralytics.models.rtdetr import RTDETR
    except Exception as e:
        print('Failed to import RTDETR from local ultralytics package:', e)
        raise


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


def xywh_to_xyxy(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw/2
    y1 = cy - bh/2
    x2 = cx + bw/2
    y2 = cy + bh/2
    return [x1, y1, x2, y2]


def draw_boxes(img, boxes, color=(255,0,0), labels=None):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    for b in boxes:
        x1,y1,x2,y2 = b[0:4]
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        if len(b) > 4 and labels is not None:
            draw.text((x1, y1-10), labels(b), fill=color, font=font)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default='/home/cjj/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml')
    parser.add_argument('--out', default='runs/detect/visual_val')
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    data = load_dataset_yaml(args.data)
    val_dir = Path(data['path']) / data['val']
    label_dir = Path(data['path']) / data['labels'] if 'labels' in data else (Path(data['path']) / data.get('labels_dir', 'labels'))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RTDETR(args.weights)

    img_files = sorted(list(val_dir.glob('*.jpg')))[:args.n]

    for img_path in img_files:
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        label_path = (Path(str(img_path).replace(str(val_dir), str(label_dir)))).with_suffix('.txt')
        gt_boxes = load_labels_for_image(label_path)
        gt_xyxy = [xywh_to_xyxy(x,y,w,h,img_w,img_h) for (_,x,y,w,h) in gt_boxes]

        # predict
        preds = model.predict(source=str(img_path), conf=0.001, imgsz=640)
        # Ultralytics predict can return a Results object, a list of Results, or tuple of tensors.
        pred_boxes = []
        res = None
        # normalize to a single result object 'res'
        if hasattr(preds, 'boxes'):
            res = preds
        elif isinstance(preds, (list, tuple)) and len(preds) > 0:
            # often a list of Results
            res = preds[0]

        if res is None:
            print('prediction parse error: unexpected return type', type(preds))
        else:
            boxes = getattr(res, 'boxes', None)
            if boxes is not None:
                for b in boxes[:20]:
                    try:
                        xyxy = b.xyxy.tolist() if hasattr(b, 'xyxy') else list(b[:4])
                        conf = float(b.conf) if hasattr(b, 'conf') else (float(b[4]) if len(b) > 4 else 0.0)
                        cls = int(b.cls) if hasattr(b, 'cls') else (int(b[5]) if len(b) > 5 else -1)
                        pred_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls))
                    except Exception:
                        continue
            else:
                # try fallback: res as tuple/list of tensors (boxes, scores, classes)
                try:
                    if isinstance(res, (list, tuple)) and len(res) >= 1:
                        boxes_t = res[0]
                        scores_t = res[1] if len(res) > 1 else None
                        classes_t = res[2] if len(res) > 2 else None
                        for i in range(min(20, int(getattr(boxes_t, 'shape', [0])[0]))):
                            xyxy = [float(x) for x in boxes_t[i].tolist()]
                            conf = float(scores_t[i]) if scores_t is not None else 0.0
                            cls = int(classes_t[i]) if classes_t is not None else -1
                            pred_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls))
                except Exception as e:
                    print('prediction parse error (fallback):', e)

        out_img = img.copy()
        draw = ImageDraw.Draw(out_img)
        # draw GT in green
        for xy in gt_xyxy:
            draw.rectangle(xy, outline=(0,255,0), width=2)
        # draw preds in red with conf
        for (x1,y1,x2,y2,conf,cls) in pred_boxes:
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
            draw.text((x1, y1-10), f'{cls}:{conf:.2f}', fill=(255,0,0))

        out_path = out_dir / img_path.name
        out_img.save(out_path)
        print('Saved', out_path)

    print('Done')
