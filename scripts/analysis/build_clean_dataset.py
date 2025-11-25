#!/usr/bin/env python3
"""
从 COCO 数据集中提取已筛选的室内图片，构建干净的 YOLO 格式数据集
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
COCO_DIR = 'datasets/coco'
COCO_ANNO_FILE = os.path.join(COCO_DIR, 'annotations/instances_train2017.json')
IMAGES_DIR = os.path.join(COCO_DIR, 'images/train2017')
FILTER_RESULT_FILE = 'dataset_filter_results/kept_image_ids.json'

OUTPUT_DIR = 'datasets/coco_indoor_clean'
OUTPUT_IMAGES = os.path.join(OUTPUT_DIR, 'images')
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, 'labels')
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

print("="*80)
print("Building Clean Indoor Dataset from COCO-Stuff Filtered Images")
print("="*80)

# 加载筛选结果
print(f"\n⏳ Loading filter results...")
with open(FILTER_RESULT_FILE, 'r') as f:
    kept_image_ids = json.load(f)

print(f"✓ Found {len(kept_image_ids)} kept images")

# 加载 COCO 注解
print(f"\n⏳ Loading COCO annotations...")
with open(COCO_ANNO_FILE, 'r') as f:
    coco_data = json.load(f)

images_info = {img['id']: img for img in coco_data['images']}
annotations_by_image = defaultdict(list)
for ann in coco_data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

category_map = {cat['id']: cat for cat in coco_data['categories']}
print(f"✓ Loaded COCO with {len(category_map)} classes")

# 创建室内特定的类别映射（只用与室内相关的类别）
INDOOR_CATEGORIES = {
    'person', 'bed', 'chair', 'couch', 'potted plant', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'teddy bear', 'cup', 'bottle', 'wine glass', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'fork', 'knife', 'spoon', 'bowl',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bicycle', 'motorcycle', 'car', 'truck', 'bus', 'train', 'airplane'
}

# 重新映射类别 ID
indoor_class_map = {}
yolo_class_id = 0
for cat_id, cat in category_map.items():
    if cat['name'] in INDOOR_CATEGORIES:
        indoor_class_map[cat_id] = yolo_class_id
        yolo_class_id += 1

print(f"✓ Mapped {len(indoor_class_map)} indoor classes for YOLO")

# ============ 处理图片和标签 ============
print(f"\n⏳ Processing {len(kept_image_ids)} images...")

successful = 0
skipped = 0
image_ids_processed = []

for idx, img_id in enumerate(kept_image_ids):
    if idx % 20 == 0:
        print(f"  Progress: {idx}/{len(kept_image_ids)}")
    
    # 检查图片是否存在
    if img_id not in images_info:
        print(f"⚠️  Image ID {img_id} not found in COCO")
        skipped += 1
        continue
    
    img_info = images_info[img_id]
    src_img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
    
    if not os.path.exists(src_img_path):
        print(f"⚠️  Image file not found: {src_img_path}")
        skipped += 1
        continue
    
    # 复制图片
    new_filename = f"{img_id:012d}.jpg"
    dst_img_path = os.path.join(OUTPUT_IMAGES, new_filename)
    shutil.copy2(src_img_path, dst_img_path)
    
    # 生成 YOLO 格式标签
    annotations = annotations_by_image.get(img_id, [])
    
    img_height = img_info['height']
    img_width = img_info['width']
    
    yolo_annotations = []
    for ann in annotations:
        cat_id = ann['category_id']
        
        # 只保留室内类别
        if cat_id not in indoor_class_map:
            continue
        
        yolo_class_id = indoor_class_map[cat_id]
        
        # 转换 COCO bbox (x, y, w, h) 为 YOLO (x_center, y_center, w_norm, h_norm)
        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # 限制在 [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    # 写入标签文件
    label_filename = f"{img_id:012d}.txt"
    label_path = os.path.join(OUTPUT_LABELS, label_filename)
    
    with open(label_path, 'w') as f:
        if yolo_annotations:
            f.write('\n'.join(yolo_annotations))
    
    successful += 1
    image_ids_processed.append(img_id)

print(f"\n✅ Processing complete:")
print(f"   ✓ Successful: {successful}")
print(f"   ⚠️  Skipped: {skipped}")

# ============ 生成类别映射文件 ============
class_names = {}
for cat_id in sorted(indoor_class_map.keys()):
    yolo_id = indoor_class_map[cat_id]
    class_names[yolo_id] = category_map[cat_id]['name']

# 生成 YAML 配置
yaml_content = f"""# YOLO dataset config - Clean Indoor COCO
path: {os.path.abspath(OUTPUT_DIR)}
train: images
val: images

nc: {len(class_names)}
names: {{{', '.join(str(i) + ': ' + repr(class_names[i]) for i in sorted(class_names.keys()))}}}
"""

yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n✅ Generated configuration:")
print(f"   • {OUTPUT_IMAGES} ({successful} images)")
print(f"   • {OUTPUT_LABELS} ({successful} labels)")
print(f"   • {yaml_path}")

# 保存元信息
meta = {
    'total_kept': len(kept_image_ids),
    'successful_processed': successful,
    'skipped': skipped,
    'image_ids': image_ids_processed,
    'class_count': len(class_names),
    'classes': class_names
}

meta_path = os.path.join(OUTPUT_DIR, 'meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"   • {meta_path}")

print(f"\n📊 Dataset Statistics:")
print(f"   Classes: {len(class_names)}")
print(f"   Images: {successful}")
print(f"   Sample classes: {', '.join(list(class_names.values())[:5])}...")

print(f"\n🚀 Ready for training!")
print(f"   Use: python train.py --data {yaml_path}")
EOF
