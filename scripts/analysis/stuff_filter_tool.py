#!/usr/bin/env python3
"""
基于 COCO-Stuff 2017 的室内场景筛选工具
利用 scene 类别信息自动识别室内/户外图片
然后支持交互式逐张审查
"""

import os
import json
import cv2
import base64
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
COCO_DIR = 'datasets/coco'
COCO_ANNO_FILE = os.path.join(COCO_DIR, 'annotations/instances_train2017.json')
IMAGES_DIR = os.path.join(COCO_DIR, 'images/train2017')
STUFF_ANNO_FILE = os.path.join(COCO_DIR, 'annotations/annotations/stuff_train2017.json')
OUTPUT_DIR = 'dataset_filter_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 室内场景关键词
INDOOR_SCENE_KEYWORDS = {
    'bedroom', 'kitchen', 'bathroom', 'living room', 'office', 'classroom',
    'dining room', 'hallway', 'staircase', 'lobby', 'store', 'restaurant',
    'inside', 'indoor', 'interior', 'shop', 'closet', 'laundry room',
    'basement', 'garage', 'attic', 'cellar', 'factory', 'warehouse',
    'gym', 'fitness', 'hospital', 'clinic', 'library', 'museum',
    'theater', 'cinema', 'conference', 'convention', 'airport',
    'train station', 'bus station', 'subway', 'terminal', 'hotel',
    'bar', 'pub', 'cafe', 'coffee', 'church', 'temple', 'mosque',
    'court', 'courtroom', 'jail', 'prison', 'police', 'station',
}

OUTDOOR_SCENE_KEYWORDS = {
    'street', 'road', 'highway', 'sidewalk', 'parking', 'lot',
    'beach', 'sea', 'ocean', 'lake', 'river', 'water',
    'forest', 'tree', 'grass', 'field', 'meadow', 'mountain',
    'sky', 'cloud', 'sunset', 'sunrise', 'weather',
    'park', 'garden', 'yard', 'playground', 'outdoor', 'outside',
    'car', 'truck', 'motorcycle', 'bicycle', 'vehicle',
    'people', 'crowd', 'pedestrian', 'walking', 'street',
}

print("="*80)
print("COCO-Stuff Based Indoor Scene Filter")
print("="*80)

# 加载 COCO 注解
print(f"\n⏳ Loading COCO annotations: {COCO_ANNO_FILE}")
with open(COCO_ANNO_FILE, 'r') as f:
    coco_data = json.load(f)

category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
images_info = {img['id']: img for img in coco_data['images']}
annotations_by_image = defaultdict(list)

for ann in coco_data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

print(f"✓ Loaded COCO with {len(category_map)} classes")
print(f"✓ Total {len(images_info)} images and {len(coco_data['annotations'])} annotations")

# 加载 COCO-Stuff 信息
stuff_info = {}
if os.path.exists(STUFF_ANNO_FILE):
    print(f"\n⏳ Loading COCO-Stuff: {STUFF_ANNO_FILE}")
    with open(STUFF_ANNO_FILE, 'r') as f:
        stuff_data = json.load(f)
    
    for img_stuff in stuff_data.get('images', []):
        img_id = img_stuff['id']
        stuff_info[img_id] = {
            'scene': img_stuff.get('scene', '').lower(),
            'supercategories': img_stuff.get('supercategories', [])
        }
    print(f"✓ Loaded COCO-Stuff for {len(stuff_info)} images")
else:
    print(f"\n⚠️  COCO-Stuff file not found at {STUFF_ANNO_FILE}")

# ============ 辅助函数 ============
def get_image_with_boxes(img_path, annotations, category_map):
    """读取图片并绘制 BBox"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    
    for idx, ann in enumerate(annotations):
        x, y, bw, bh = ann['bbox']
        x, y = int(x), int(y)
        bw, bh = int(bw), int(bh)
        x, y = max(0, x), max(0, y)
        x_end, y_end = min(w, x + bw), min(h, y + bh)
        
        cat_id = ann['category_id']
        cat_name = category_map.get(cat_id, f'Class_{cat_id}')
        
        color = colors[idx % len(colors)]
        cv2.rectangle(img, (x, y), (x_end, y_end), color, 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(cat_name, font, 0.6, 2)
        cv2.rectangle(img, (x, max(0, y-text_h-8)), (x+text_w+10, y), color, -1)
        cv2.putText(img, cat_name, (x+5, y-4), font, 0.6, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode()

# ============ 分类图片 ============
print("\n" + "="*80)
print("Analyzing images by scene type...")
print("="*80 + "\n")

indoor_images = []
outdoor_images = []
unknown_images = []

# 处理前 15000 张以加快速度
total_to_process = min(15000, len(images_info))
processed = 0

for img_id in sorted(list(images_info.keys())[:total_to_process]):
    img_info = images_info[img_id]
    filename = img_info['file_name']
    img_path = os.path.join(IMAGES_DIR, filename)
    
    if not os.path.exists(img_path):
        continue
    
    annotations = annotations_by_image.get(img_id, [])
    if not annotations:
        continue
    
    processed += 1
    if processed % 1000 == 0:
        print(f"  Processed {processed} images...")
    
    scene_type = 'unknown'
    confidence = 0.0
    scene_name = ''
    
    # 使用 COCO-Stuff 信息
    if img_id in stuff_info:
        scene_name = stuff_info[img_id].get('scene', '')
        
        if any(indoor_kw in scene_name for indoor_kw in INDOOR_SCENE_KEYWORDS):
            scene_type = 'indoor'
            confidence = 0.95
        elif any(outdoor_kw in scene_name for outdoor_kw in OUTDOOR_SCENE_KEYWORDS):
            scene_type = 'outdoor'
            confidence = 0.95
    
    img_data = {
        'image_id': img_id,
        'filename': filename,
        'scene_type': scene_type,
        'scene_name': scene_name,
        'confidence': confidence,
        'annotation_count': len(annotations),
        'classes': list(set(category_map.get(ann['category_id'], f'Class_{ann["category_id"]}') 
                          for ann in annotations))
    }
    
    if scene_type == 'indoor':
        indoor_images.append(img_data)
    elif scene_type == 'outdoor':
        outdoor_images.append(img_data)
    else:
        unknown_images.append(img_data)

print(f"\n📊 Scene Type Distribution:")
total_classified = len(indoor_images) + len(outdoor_images) + len(unknown_images)
print(f"   🏠 Indoor:  {len(indoor_images):6d} ({len(indoor_images)*100//total_classified:2d}%)")
print(f"   🌳 Outdoor: {len(outdoor_images):6d} ({len(outdoor_images)*100//total_classified:2d}%)")
print(f"   ❓ Unknown: {len(unknown_images):6d} ({len(unknown_images)*100//total_classified:2d}%)")

# ============ 生成筛选工具 ============
print("\n⏳ Generating filter HTML...")
print("  Processing images with BBox...")

filter_images = []

# 先审查 unknown 的前 100 张
for i, img_data in enumerate(unknown_images[:100]):
    if i % 20 == 0:
        print(f"    Unknown: {i}/100")
    
    img_id = img_data['image_id']
    filename = img_data['filename']
    img_path = os.path.join(IMAGES_DIR, filename)
    
    annotations = annotations_by_image[img_id]
    img_base64 = get_image_with_boxes(img_path, annotations, category_map)
    
    if img_base64:
        filter_images.append({
            'image_id': img_id,
            'filename': filename,
            'image': img_base64,
            'scene_type': img_data['scene_type'],
            'scene_name': img_data.get('scene_name', 'unknown'),
            'confidence': img_data['confidence'],
            'annotation_count': len(annotations),
            'classes': img_data['classes'],
            'auto_label': f"❓ Unknown - Scene: {img_data.get('scene_name', 'N/A')}"
        })

# 加入一些 outdoor 的作为参考
for i, img_data in enumerate(outdoor_images[:50]):
    if i % 20 == 0:
        print(f"    Outdoor: {i}/50")
    
    img_id = img_data['image_id']
    filename = img_data['filename']
    img_path = os.path.join(IMAGES_DIR, filename)
    
    annotations = annotations_by_image[img_id]
    img_base64 = get_image_with_boxes(img_path, annotations, category_map)
    
    if img_base64:
        filter_images.append({
            'image_id': img_id,
            'filename': filename,
            'image': img_base64,
            'scene_type': img_data['scene_type'],
            'scene_name': img_data.get('scene_name', 'outdoor'),
            'confidence': img_data['confidence'],
            'annotation_count': len(annotations),
            'classes': img_data['classes'],
            'auto_label': f"🌳 Outdoor - Scene: {img_data.get('scene_name', 'N/A')}"
        })

print(f"\n✓ Prepared {len(filter_images)} images for review\n")

# 生成 HTML
html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>COCO-Stuff Scene Filter - Indoor Detection</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        header {{
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }}
        h1 {{ color: #333; margin-bottom: 10px; font-size: 32px; }}
        .subtitle {{ color: #666; font-size: 14px; }}
        .progress-bar {{
            background: #e0e0e0;
            height: 12px;
            border-radius: 6px;
            margin: 20px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
        }}
        .stats-box {{
            background: #f0f7ff;
            border: 2px solid #667eea;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
            line-height: 1.8;
        }}
        .stats-box strong {{ color: #667eea; }}
        .content {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .image-container {{
            background: #f5f5f5;
            border: 3px solid #ddd;
            border-radius: 8px;
            aspect-ratio: 16/12;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .info-section {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .info-box {{
            background: #fafafa;
            padding: 15px;
            border-radius: 6px;
            border: 2px solid #e0e0e0;
        }}
        .info-row {{
            margin: 8px 0;
            font-size: 13px;
        }}
        .label {{ color: #666; font-weight: 600; }}
        .value {{ color: #333; margin-left: 5px; }}
        .auto-label {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
        }}
        .indoor-label {{
            background: #c8e6c9;
            color: #1b5e20;
        }}
        .outdoor-label {{
            background: #ffe0b2;
            color: #e65100;
        }}
        .unknown-label {{
            background: #f3e5f5;
            color: #4a148c;
        }}
        .objects-box {{
            background: #f0f7ff;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .objects-box h4 {{ margin-bottom: 12px; color: #333; }}
        .object-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .object-tag {{
            background: white;
            border: 2px solid #667eea;
            color: #667eea;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .controls {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        .btn {{
            padding: 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .btn-keep {{
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
        }}
        .btn-keep:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(76, 175, 80, 0.4);
        }}
        .btn-remove {{
            background: linear-gradient(135deg, #F44336 0%, #C62828 100%);
            color: white;
        }}
        .btn-remove:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(244, 67, 54, 0.4);
        }}
        .nav-bar {{
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .nav-btn {{
            padding: 10px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .nav-btn:hover {{ background: #764ba2; }}
        .nav-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        #completion {{
            background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
            padding: 30px;
            border-radius: 8px;
            text-align: center;
            margin-top: 30px;
            display: none;
        }}
        #completion h2 {{ color: #1b5e20; margin-bottom: 15px; }}
        #completion p {{ color: #2e7d32; font-size: 14px; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏠 COCO-Stuff Scene Filter</h1>
            <p class="subtitle">Indoor vs Outdoor Classification - Scene-based Filtering</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progress"></div>
            </div>
        </header>

        <div class="stats-box">
            <strong>📊 Dataset Analysis:</strong><br>
            🏠 Indoor: {len(indoor_images):,} images | 
            🌳 Outdoor: {len(outdoor_images):,} images | 
            ❓ Unknown: {len(unknown_images):,} images<br>
            <strong>Current task:</strong> Review {len(filter_images)} images for scene type accuracy
        </div>

        <div class="content">
            <div>
                <div class="image-container">
                    <img id="image" src="" alt="Loading...">
                </div>
            </div>

            <div class="info-section">
                <div class="info-box">
                    <div class="info-row">
                        <span class="label">Image:</span>
                        <span class="value" id="filename">-</span>
                    </div>
                    <div class="info-row">
                        <span class="label">Scene:</span>
                        <span class="value" id="scenename">-</span>
                    </div>
                    <div class="info-row">
                        <span class="label">Status:</span>
                        <span class="value" id="status">⏳ Pending</span>
                    </div>
                    <div class="info-row">
                        <span class="label">Progress:</span>
                        <span class="value" id="progress-text">0 / {len(filter_images)}</span>
                    </div>
                    <span class="auto-label" id="autoLabel"></span>
                </div>

                <div class="objects-box">
                    <h4>📋 Objects Detected:</h4>
                    <div class="object-tags" id="objects"></div>
                </div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-keep" onclick="submitDecision('keep')">✅ YES - Indoor</button>
            <button class="btn btn-remove" onclick="submitDecision('remove')">❌ NO - Not Indoor</button>
        </div>

        <div class="nav-bar">
            <button class="nav-btn" id="prevBtn" onclick="previousImage()">← Previous</button>
            <button class="nav-btn" id="nextBtn" onclick="nextImage()">Next →</button>
            <button class="nav-btn" onclick="downloadResults()">💾 Download Results</button>
            <span style="margin-left: auto; font-size: 12px; color: #666;">
                Keyboard: ← → for navigation, K to keep, R to remove
            </span>
        </div>

        <div id="completion">
            <h2>✅ Review Complete!</h2>
            <p id="completionText"></p>
        </div>
    </div>

    <script>
        const imagesData = {json.dumps(filter_images)};
        let currentIdx = 0;
        let decisions = JSON.parse(localStorage.getItem('stuff_filter_decisions') || '{{}}');
        
        function displayImage() {{
            if (currentIdx >= imagesData.length) return;
            
            const data = imagesData[currentIdx];
            document.getElementById('image').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('filename').textContent = data.filename;
            document.getElementById('scenename').textContent = data.scene_name || 'Unknown';
            document.getElementById('progress-text').textContent = 
                (currentIdx + 1) + ' / ' + imagesData.length;
            document.getElementById('progress').style.width = 
                ((currentIdx + 1) / imagesData.length * 100) + '%';
            
            // Auto label
            let labelClass = 'unknown-label';
            if (data.scene_type === 'indoor') labelClass = 'indoor-label';
            else if (data.scene_type === 'outdoor') labelClass = 'outdoor-label';
            
            document.getElementById('autoLabel').innerHTML = 
                `<strong>${{data.auto_label}}</strong>`;
            document.getElementById('autoLabel').className = 'auto-label ' + labelClass;
            
            // Status
            if (String(currentIdx) in decisions) {{
                const dec = decisions[currentIdx];
                document.getElementById('status').textContent = 
                    dec === 'keep' ? '✅ KEPT' : '❌ REMOVED';
                document.getElementById('status').style.color = 
                    dec === 'keep' ? '#4CAF50' : '#F44336';
            }} else {{
                document.getElementById('status').textContent = '⏳ Pending';
                document.getElementById('status').style.color = '#FF9800';
            }}
            
            // Objects
            const objectsHtml = data.classes.map(cls => 
                `<span class="object-tag">${{cls}}</span>`
            ).join('');
            document.getElementById('objects').innerHTML = objectsHtml || '<p style="color: #999;">No objects</p>';
            
            document.getElementById('prevBtn').disabled = currentIdx === 0;
            document.getElementById('nextBtn').disabled = currentIdx === imagesData.length - 1;
        }}
        
        function submitDecision(decision) {{
            decisions[currentIdx] = decision;
            localStorage.setItem('stuff_filter_decisions', JSON.stringify(decisions));
            
            if (currentIdx < imagesData.length - 1) {{
                currentIdx++;
                displayImage();
            }} else {{
                showCompletion();
            }}
        }}
        
        function nextImage() {{
            if (currentIdx < imagesData.length - 1) {{
                currentIdx++;
                displayImage();
            }}
        }}
        
        function previousImage() {{
            if (currentIdx > 0) {{
                currentIdx--;
                displayImage();
            }}
        }}
        
        function downloadResults() {{
            const results = {{
                task: 'COCO-Stuff scene type validation',
                timestamp: new Date().toISOString(),
                total_reviewed: Object.keys(decisions).length,
                kept: Object.values(decisions).filter(d => d === 'keep').length,
                removed: Object.values(decisions).filter(d => d === 'remove').length,
                results: imagesData.map((img, idx) => ({{
                    index: idx,
                    image_id: img.image_id,
                    filename: img.filename,
                    scene_name: img.scene_name,
                    auto_label: img.auto_label,
                    decision: decisions[idx] || 'pending'
                }}))
            }};
            
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'stuff_filter_results_' + Date.now() + '.json';
            link.click();
            URL.revokeObjectURL(url);
        }}
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') previousImage();
            else if (e.key === 'ArrowRight') nextImage();
            else if (e.key.toLowerCase() === 'k') submitDecision('keep');
            else if (e.key.toLowerCase() === 'r') submitDecision('remove');
        }});
        
        function showCompletion() {{
            const kept = Object.values(decisions).filter(d => d === 'keep').length;
            const removed = Object.values(decisions).filter(d => d === 'remove').length;
            document.getElementById('completionText').innerHTML = 
                `<strong>Kept:</strong> ${{kept}} | <strong>Removed:</strong> ${{removed}}<br><br>All decisions saved to browser!`;
            document.getElementById('completion').style.display = 'block';
        }}
        
        displayImage();
    </script>
</body>
</html>
'''

html_path = os.path.join(OUTPUT_DIR, 'stuff_filter.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

# 保存统计信息
stats_path = os.path.join(OUTPUT_DIR, 'scene_analysis.json')
with open(stats_path, 'w') as f:
    json.dump({
        'total_images': len(images_info),
        'processed_images': processed,
        'indoor_images': len(indoor_images),
        'outdoor_images': len(outdoor_images),
        'unknown_images': len(unknown_images),
        'filter_images_count': len(filter_images),
        'indoor_image_ids': [img['image_id'] for img in indoor_images],
        'outdoor_image_ids': [img['image_id'] for img in outdoor_images],
        'unknown_image_ids': [img['image_id'] for img in unknown_images]
    }, f, indent=2)

print("="*80)
print("✅ Filter tool generated!")
print("="*80)
print(f"\n📄 Output files:")
print(f"   • {html_path}")
print(f"   • {stats_path}")
print(f"\n📂 To use:")
print(f"   cd {OUTPUT_DIR}")
print(f"   python3 -m http.server 8888")
print(f"   Open browser: http://localhost:8888/stuff_filter.html")
print(f"\n⌨️  Keyboard shortcuts:")
print(f"   ← → Navigate | K Keep | R Remove")
