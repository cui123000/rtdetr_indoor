#!/usr/bin/env python3
"""
下载 COCO-Stuff 2017 场景标注
"""

import os
import urllib.request
import zipfile
import json

ANNO_DIR = 'datasets/coco/annotations'
os.makedirs(ANNO_DIR, exist_ok=True)

# COCO-Stuff 2017 annotations URL
STUFF_URL = 'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip'
ZIP_PATH = os.path.join(ANNO_DIR, 'stuff_annotations.zip')
STUFF_JSON = os.path.join(ANNO_DIR, 'stuff_train2017.json')

print("Downloading COCO-Stuff annotations...")
print(f"URL: {STUFF_URL}\n")

try:
    urllib.request.urlretrieve(STUFF_URL, ZIP_PATH, 
        lambda block_count, block_size, total_size: 
        print(f"\r⏳ {block_count * block_size / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB", end=''))
    
    print("\n✅ Downloaded\n")
    
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(ANNO_DIR)
    
    print("✅ Extracted\n")
    
    # 检查提取的文件
    extracted = [f for f in os.listdir(ANNO_DIR) if 'stuff' in f.lower() and f.endswith('.json')]
    print(f"Found files: {extracted}")
    
    if extracted:
        print(f"\n✅ COCO-Stuff annotations ready at {ANNO_DIR}/")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
