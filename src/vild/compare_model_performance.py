#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ViLD-RTDETR 与 原始 RTDETR 性能对比评估脚本
此脚本用于比较两个模型的性能差异和提升指标
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# 添加安全的全局变量，允许加载 ultralytics 模型
try:
    import torch.serialization
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
except (ImportError, AttributeError):
    print("⚠️ 无法添加安全全局变量，将尝试使用 weights_only=False 加载模型")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加项目根目录到模块搜索路径
sys.path.append(project_root)
# 不添加RTDETR路径，避免导入问题
# sys.path.append(os.path.join(project_root, 'src/RT-DETR/rtdetr_pytorch'))

def compare_models(image_path, output_dir=None, conf_threshold=0.35):
    """比较ViLD-RTDETR模型与原始RTDETR模型的性能"""
    from src.vild.vild_modular.config import MODEL_CONFIG, INFERENCE_CONFIG
    from src.vild.vild_modular.detector import FixedViLDDetector
    from src.vild.vild_modular.scene_classifier import SceneClassifier
    from src.vild.vild_modular.model import load_models
    
    # 定义COCO类别名称，避免导入RTDETR的类别
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(current_dir, "model_comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 结果将保存在: {output_dir}")
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 加载ViLD-RTDETR模型
    print("📦 正在加载ViLD-RTDETR模型...")
    rtdetr_model, clip_model, image_processor, clip_preprocess = load_models(
        rtdetr_path=MODEL_CONFIG['rtdetr_model_path'],
        clip_name=MODEL_CONFIG['clip_model_name'],
        device=device
    )
    
    # 创建场景分类器
    scene_classifier = SceneClassifier(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device
    )
    
    # 创建ViLD检测器
    vild_detector = FixedViLDDetector(
        clip_model=clip_model,
        detector_model=rtdetr_model,
        image_processor=image_processor,
        clip_preprocess=clip_preprocess,
        device=device,
        projector_path=MODEL_CONFIG['projector_path'],
        config={'model': MODEL_CONFIG, 'inference': INFERENCE_CONFIG}
    )
    
    # 加载原始RTDETR模型
    print("📦 正在加载原始RTDETR模型...")
    try:
        # 使用 transformers 库直接加载 RTDETR 模型
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        
        rtdetr_orig_path = os.path.join(project_root, "rtdetr-l.pt")
        
        if os.path.exists(rtdetr_orig_path):
            # 创建原始模型实例
            print("📄 找到原始 RTDETR 模型权重文件")
            print("🔄 加载原始 RTDETR 模型")
            
            # 尝试直接使用 torch.hub.load 加载模型
            try:
                # 直接从本地加载
                print(f"尝试使用 torch.hub.load 加载模型")
                rtdetr_orig = torch.hub.load('ultralytics/yolov5', 'custom', path=rtdetr_orig_path, device=device)
                rtdetr_orig.eval()
                rtdetr_orig_processor = None  # YOLOv5不需要单独的处理器
                print("✅ 使用 torch.hub.load 成功加载模型")
            except Exception as e:
                print(f"⚠️ torch.hub.load 加载失败: {e}")
                print(f"尝试使用 transformers 加载模型")
                
                # 备用方案：使用 transformers 加载模型
                rtdetr_orig = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)
                
                # 使用 weights_only=False 加载模型权重，解决 PyTorch 2.6 中的限制
                try:
                    model_weights = torch.load(rtdetr_orig_path, map_location=device, weights_only=False)
                    rtdetr_orig.load_state_dict(model_weights)
                    print("✅ 使用 weights_only=False 成功加载模型")
                except Exception as e2:
                    print(f"⚠️ weights_only=False 加载失败: {e2}")
                    # 最后尝试 context manager 方法
                    try:
                        with torch.serialization.safe_globals(['ultralytics.nn.tasks.DetectionModel']):
                            model_weights = torch.load(rtdetr_orig_path, map_location=device)
                        rtdetr_orig.load_state_dict(model_weights)
                        print("✅ 使用 safe_globals context manager 成功加载模型")
                    except Exception as e3:
                        print(f"❌ 所有加载方法都失败: {e3}")
                        raise e3
                
                rtdetr_orig.eval()
                rtdetr_orig_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
            
            print("✅ 原始RTDETR模型加载成功")
            has_rtdetr = True
        else:
            print(f"⚠️ 找不到原始RTDETR模型权重文件: {rtdetr_orig_path}")
            print("⚠️ 将仅评估ViLD-RTDETR模型")
            has_rtdetr = False
    except Exception as e:
        print(f"❌ 无法加载原始RTDETR模型: {e}")
        print("⚠️ 将仅评估ViLD-RTDETR模型")
        has_rtdetr = False
    
    # 确定输入是单张图像还是目录
    if os.path.isdir(image_path):
        print(f"🔍 批量评估图像目录: {image_path}")
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ 找到 {len(image_files)} 个图像文件")
    else:
        print(f"🔍 评估单张图像: {image_path}")
        image_files = [image_path]
    
    # 准备结果统计
    vild_results = []
    rtdetr_results = []
    vild_total_time = 0
    rtdetr_total_time = 0
    vild_category_counts = {}
    rtdetr_category_counts = {}
    comparison_metrics = {
        'total_images': len(image_files),
        'per_image_comparison': [],
        'category_comparison': {},
        'average_metrics': {
            'vild_avg_detections': 0,
            'rtdetr_avg_detections': 0,
            'vild_avg_time': 0,
            'rtdetr_avg_time': 0,
            'vild_fps': 0,
            'rtdetr_fps': 0,
            'detection_increase_percent': 0,
            'speed_difference_percent': 0
        }
    }
    
    # 图像转换器
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 处理每个图像
    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            # 创建比较结果图像
            comparison_image = Image.new('RGB', (img_width*2, img_height), (255, 255, 255))
            comparison_image.paste(image, (0, 0))
            comparison_image.paste(image, (img_width, 0))
            
            # 绘制两个模型的结果
            draw = ImageDraw.Draw(comparison_image)
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("Arial.ttf", 15)
            except:
                font = ImageFont.load_default()
            
            # 1. 原始RTDETR模型检测
            rtdetr_image_detections = []
            if has_rtdetr:
                rtdetr_start_time = time.time()
                
                # 检查是否是 YOLOv5 模型
                is_yolov5 = rtdetr_orig_processor is None
                
                if is_yolov5:
                    # YOLOv5 方式处理
                    img_np = np.array(image)
                    
                    # 推理
                    with torch.no_grad():
                        results = rtdetr_orig(img_np)
                        
                    rtdetr_end_time = time.time()
                    rtdetr_process_time = rtdetr_end_time - rtdetr_start_time
                    rtdetr_total_time += rtdetr_process_time
                    
                    # 处理 YOLOv5 结果
                    boxes = []
                    scores = []
                    labels = []
                    
                    # 提取 YOLOv5 结果
                    if hasattr(results, 'xyxy'):
                        # YOLOv5 结果格式
                        yolo_results = results.xyxy[0].cpu().numpy()
                        for det in yolo_results:
                            x1, y1, x2, y2, conf, cls = det
                            if conf >= conf_threshold:
                                boxes.append([x1, y1, x2, y2])
                                scores.append(conf)
                                labels.append(int(cls))
                    elif hasattr(results, 'pandas') and callable(getattr(results, 'pandas')):
                        # 另一种 YOLOv5 格式
                        df = results.pandas().xyxy[0]
                        for _, row in df.iterrows():
                            if row['confidence'] >= conf_threshold:
                                boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                                scores.append(row['confidence'])
                                labels.append(row['class'])
                    else:
                        # 直接处理 results
                        try:
                            for det in results:
                                for *xyxy, conf, cls in det:
                                    if conf >= conf_threshold:
                                        x1, y1, x2, y2 = [float(val) for val in xyxy]
                                        boxes.append([x1, y1, x2, y2])
                                        scores.append(float(conf))
                                        labels.append(int(cls))
                        except Exception as e:
                            print(f"⚠️ 无法解析 YOLOv5 结果: {e}")
                    
                    # 转换为 numpy 数组
                    if boxes:
                        boxes = np.array(boxes)
                        scores = np.array(scores)
                        labels = np.array(labels)
                    else:
                        boxes = np.array([])
                        scores = np.array([])
                        labels = np.array([])
                else:
                    # Transformers 方式处理
                    # 使用 transformers 的处理器预处理图像
                    inputs = rtdetr_orig_processor(images=image, return_tensors="pt").to(device)
                    
                    # 推理
                    with torch.no_grad():
                        outputs = rtdetr_orig(**inputs)
                    
                    rtdetr_end_time = time.time()
                    rtdetr_process_time = rtdetr_end_time - rtdetr_start_time
                    rtdetr_total_time += rtdetr_process_time
                    
                    # 处理检测结果 - transformers格式
                    results = rtdetr_orig_processor.post_process_object_detection(outputs, threshold=conf_threshold)
                    boxes = results[0]['boxes'].cpu().numpy()
                    scores = results[0]['scores'].cpu().numpy()
                    labels = results[0]['labels'].cpu().numpy()
                
                # 由于post_process_object_detection已经过滤了低置信度结果，这里直接使用
                valid_boxes = boxes
                valid_scores = scores
                valid_labels = labels
                
                # 更新类别计数
                for label_idx in valid_labels:
                    category_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"
                    rtdetr_category_counts[category_name] = rtdetr_category_counts.get(category_name, 0) + 1
                
                # 绘制检测框 - 右侧图像
                for box, score, label_idx in zip(valid_boxes, valid_scores, valid_labels):
                    # 随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    # 获取类别名称
                    category_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"
                    
                    # 绘制边界框 - 向右偏移img_width
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1+img_width, y1, x2+img_width, y2], outline=color, width=2)
                    
                    # 绘制标签
                    label = f"{category_name} {score:.2f}"
                    draw.rectangle([x1+img_width, y1, x1+img_width + len(label) * 8, y1 + 15], fill=color)
                    draw.text((x1+img_width, y1), label, fill="white", font=font)
                    
                    # 保存检测结果
                    rtdetr_image_detections.append({
                        'bbox': box.tolist(),
                        'score': float(score),
                        'category': category_name
                    })
                
                # 在右上角添加RTDETR信息
                rtdetr_info = f"原始RTDETR: {len(valid_boxes)}个物体, {rtdetr_process_time:.3f}s ({1/rtdetr_process_time:.1f} FPS)"
                draw.rectangle([img_width, 0, img_width + len(rtdetr_info) * 8, 20], fill="blue")
                draw.text((img_width+5, 5), rtdetr_info, fill="white", font=font)
            
            # 2. ViLD-RTDETR模型检测
            vild_start_time = time.time()
            
            # 场景分类
            scene_type, scene_score = scene_classifier.classify_scene(image)
            
            # 物体检测
            detection_result = vild_detector.detect_objects(
                image, 
                scene_type=scene_type,
                use_macro_categories=True  # 使用大类别分组
            )
            
            vild_end_time = time.time()
            vild_process_time = vild_end_time - vild_start_time
            vild_total_time += vild_process_time
            
            # 处理检测结果
            vild_image_detections = []
            if detection_result and 'boxes' in detection_result and len(detection_result['boxes']) > 0:
                boxes = detection_result['boxes']
                scores = detection_result['scores']
                categories = detection_result['labels']
                
                # 过滤低置信度结果
                valid_indices = [i for i, score in enumerate(scores) if score >= conf_threshold]
                valid_boxes = [boxes[i] for i in valid_indices]
                valid_scores = [scores[i] for i in valid_indices]
                valid_categories = [categories[i] for i in valid_indices]
                
                # 更新类别计数
                for category in valid_categories:
                    vild_category_counts[category] = vild_category_counts.get(category, 0) + 1
                
                # 绘制检测框 - 左侧图像
                for box, score, category in zip(valid_boxes, valid_scores, valid_categories):
                    # 随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    # 绘制边界框
                    if len(box) == 4:  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                    else:  # [x, y, w, h]
                        x1, y1, w, h = box
                        x2, y2 = x1 + w, y1 + h
                        
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # 绘制标签
                    label = f"{category} {score:.2f}"
                    draw.rectangle([x1, y1, x1 + len(label) * 8, y1 + 15], fill=color)
                    draw.text((x1, y1), label, fill="white", font=font)
                    
                    # 保存检测结果
                    vild_image_detections.append({
                        'bbox': box.tolist() if isinstance(box, np.ndarray) else box,
                        'score': float(score),
                        'category': category
                    })
            
            # 在左上角添加ViLD信息
            vild_info = f"ViLD-RTDETR: {len(vild_image_detections)}个物体, {vild_process_time:.3f}s ({1/vild_process_time:.1f} FPS)"
            scene_info = f"场景: {scene_type} ({scene_score:.2f})"
            draw.rectangle([0, 0, len(vild_info) * 8, 40], fill="green")
            draw.text((5, 5), vild_info, fill="white", font=font)
            draw.text((5, 22), scene_info, fill="white", font=font)
            
            # 保存比较图像
            output_file = os.path.join(output_dir, f"compare_{os.path.basename(img_path)}")
            comparison_image.save(output_file)
            
            # 计算图像级别的比较指标
            rtdetr_detections_count = len(rtdetr_image_detections) if has_rtdetr else 0
            vild_detections_count = len(vild_image_detections)
            
            detection_increase = vild_detections_count - rtdetr_detections_count
            detection_increase_percent = ((vild_detections_count / rtdetr_detections_count) - 1) * 100 if rtdetr_detections_count > 0 else float('inf')
            
            speed_difference = rtdetr_process_time - vild_process_time if has_rtdetr else 0
            speed_difference_percent = ((rtdetr_process_time / vild_process_time) - 1) * 100 if has_rtdetr and vild_process_time > 0 else float('inf')
            
            # 添加到每张图像的比较结果
            comparison_metrics['per_image_comparison'].append({
                'image_path': img_path,
                'vild_detections': vild_detections_count,
                'rtdetr_detections': rtdetr_detections_count,
                'detection_increase': detection_increase,
                'detection_increase_percent': detection_increase_percent if detection_increase_percent != float('inf') else 'N/A',
                'vild_process_time': vild_process_time,
                'rtdetr_process_time': rtdetr_process_time if has_rtdetr else 'N/A',
                'vild_fps': 1/vild_process_time if vild_process_time > 0 else 0,
                'rtdetr_fps': 1/rtdetr_process_time if has_rtdetr and rtdetr_process_time > 0 else 'N/A',
                'speed_difference': speed_difference,
                'speed_difference_percent': speed_difference_percent if speed_difference_percent != float('inf') else 'N/A',
            })
            
            # 添加到总体结果
            vild_results.append({
                'image_path': img_path,
                'scene_type': scene_type,
                'scene_score': float(scene_score),
                'process_time': vild_process_time,
                'fps': 1/vild_process_time if vild_process_time > 0 else 0,
                'detections': vild_image_detections
            })
            
            if has_rtdetr:
                rtdetr_results.append({
                    'image_path': img_path,
                    'process_time': rtdetr_process_time,
                    'fps': 1/rtdetr_process_time if rtdetr_process_time > 0 else 0,
                    'detections': rtdetr_image_detections
                })
            
            # 输出每张图像的比较结果
            print(f"\n--- 图像: {os.path.basename(img_path)} ---")
            print(f"ViLD-RTDETR: {vild_detections_count}个物体, {vild_process_time:.3f}s ({1/vild_process_time:.1f} FPS)")
            if has_rtdetr:
                print(f"原始RTDETR: {rtdetr_detections_count}个物体, {rtdetr_process_time:.3f}s ({1/rtdetr_process_time:.1f} FPS)")
                print(f"检测增量: {detection_increase}个物体 ({detection_increase_percent:.1f}%)")
                print(f"速度差异: {speed_difference:.3f}s ({speed_difference_percent:.1f}%)")
        
        except Exception as e:
            print(f"❌ 处理图像 {os.path.basename(img_path)} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算总体统计信息
    vild_avg_time = vild_total_time / len(image_files) if image_files else 0
    vild_avg_fps = 1.0 / vild_avg_time if vild_avg_time > 0 else 0
    vild_total_detections = sum(len(r['detections']) for r in vild_results)
    vild_avg_detections = vild_total_detections / len(vild_results) if vild_results else 0
    
    comparison_metrics['average_metrics']['vild_avg_detections'] = vild_avg_detections
    comparison_metrics['average_metrics']['vild_avg_time'] = vild_avg_time
    comparison_metrics['average_metrics']['vild_fps'] = vild_avg_fps
    
    if has_rtdetr:
        rtdetr_avg_time = rtdetr_total_time / len(image_files) if image_files else 0
        rtdetr_avg_fps = 1.0 / rtdetr_avg_time if rtdetr_avg_time > 0 else 0
        rtdetr_total_detections = sum(len(r['detections']) for r in rtdetr_results)
        rtdetr_avg_detections = rtdetr_total_detections / len(rtdetr_results) if rtdetr_results else 0
        
        comparison_metrics['average_metrics']['rtdetr_avg_detections'] = rtdetr_avg_detections
        comparison_metrics['average_metrics']['rtdetr_avg_time'] = rtdetr_avg_time
        comparison_metrics['average_metrics']['rtdetr_fps'] = rtdetr_avg_fps
        
        # 计算平均提升
        avg_detection_increase_percent = ((vild_avg_detections / rtdetr_avg_detections) - 1) * 100 if rtdetr_avg_detections > 0 else float('inf')
        avg_speed_difference_percent = ((rtdetr_avg_time / vild_avg_time) - 1) * 100 if vild_avg_time > 0 else float('inf')
        
        comparison_metrics['average_metrics']['detection_increase_percent'] = avg_detection_increase_percent if avg_detection_increase_percent != float('inf') else 'N/A'
        comparison_metrics['average_metrics']['speed_difference_percent'] = avg_speed_difference_percent if avg_speed_difference_percent != float('inf') else 'N/A'
    
    # 计算类别对比
    all_categories = set(vild_category_counts.keys()) | set(rtdetr_category_counts.keys())
    for category in all_categories:
        vild_count = vild_category_counts.get(category, 0)
        rtdetr_count = rtdetr_category_counts.get(category, 0)
        
        # 计算类别检测增量
        if rtdetr_count > 0:
            category_increase_percent = ((vild_count / rtdetr_count) - 1) * 100
        elif vild_count > 0:
            category_increase_percent = float('inf')  # ViLD检测到了，RTDETR没有检测到
        else:
            category_increase_percent = 0  # 两者都没有检测到
            
        comparison_metrics['category_comparison'][category] = {
            'vild_count': vild_count,
            'rtdetr_count': rtdetr_count,
            'increase': vild_count - rtdetr_count,
            'increase_percent': category_increase_percent if category_increase_percent != float('inf') else 'N/A'
        }
    
    # 保存详细结果
    vild_results_file = os.path.join(output_dir, 'vild_detection_results.json')
    with open(vild_results_file, 'w') as f:
        json.dump(vild_results, f, indent=2)
    
    if has_rtdetr:
        rtdetr_results_file = os.path.join(output_dir, 'rtdetr_detection_results.json')
        with open(rtdetr_results_file, 'w') as f:
            json.dump(rtdetr_results, f, indent=2)
    
    # 保存比较指标
    comparison_file = os.path.join(output_dir, 'comparison_metrics.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    
    # 绘制检测数量对比图
    plt.figure(figsize=(12, 6))
    
    # 提取每张图像的检测数量
    image_names = [os.path.basename(img_path) for img_path in image_files]
    vild_counts = [len(r['detections']) for r in vild_results]
    
    if has_rtdetr:
        rtdetr_counts = [len(r['detections']) for r in rtdetr_results]
        
        # 绘制双柱状图
        x = np.arange(len(image_names))
        width = 0.35
        
        plt.bar(x - width/2, vild_counts, width, label='ViLD-RTDETR')
        plt.bar(x + width/2, rtdetr_counts, width, label='原始RTDETR')
        
        plt.xlabel('图像')
        plt.ylabel('检测物体数量')
        plt.title('ViLD-RTDETR vs 原始RTDETR 检测数量对比')
        plt.xticks(x, [name[:10] + '...' if len(name) > 10 else name for name in image_names], rotation=45)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        detection_chart_file = os.path.join(output_dir, 'detection_count_comparison.png')
        plt.savefig(detection_chart_file)
        
        # 绘制FPS对比图
        plt.figure(figsize=(12, 6))
        
        vild_fps = [1/r['process_time'] if r['process_time'] > 0 else 0 for r in vild_results]
        rtdetr_fps = [1/r['process_time'] if r['process_time'] > 0 else 0 for r in rtdetr_results]
        
        plt.bar(x - width/2, vild_fps, width, label='ViLD-RTDETR')
        plt.bar(x + width/2, rtdetr_fps, width, label='原始RTDETR')
        
        plt.xlabel('图像')
        plt.ylabel('FPS (帧每秒)')
        plt.title('ViLD-RTDETR vs 原始RTDETR 速度对比')
        plt.xticks(x, [name[:10] + '...' if len(name) > 10 else name for name in image_names], rotation=45)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        fps_chart_file = os.path.join(output_dir, 'fps_comparison.png')
        plt.savefig(fps_chart_file)
        
        # 绘制类别检测数量对比图 (仅显示前10个类别)
        plt.figure(figsize=(14, 8))
        
        # 按ViLD检测数量排序
        top_categories = sorted(vild_category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        category_names = [cat for cat, _ in top_categories]
        
        vild_cat_counts = [vild_category_counts.get(cat, 0) for cat in category_names]
        rtdetr_cat_counts = [rtdetr_category_counts.get(cat, 0) for cat in category_names]
        
        x = np.arange(len(category_names))
        
        plt.bar(x - width/2, vild_cat_counts, width, label='ViLD-RTDETR')
        plt.bar(x + width/2, rtdetr_cat_counts, width, label='原始RTDETR')
        
        plt.xlabel('物体类别')
        plt.ylabel('检测数量')
        plt.title('ViLD-RTDETR vs 原始RTDETR 类别检测数量对比 (前10类)')
        plt.xticks(x, category_names, rotation=45)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        category_chart_file = os.path.join(output_dir, 'category_comparison.png')
        plt.savefig(category_chart_file)
    
    # 打印总体比较结果
    print("\n===== 模型性能对比摘要 =====")
    print(f"总图像数: {len(image_files)}")
    print("\nViLD-RTDETR模型:")
    print(f"- 总检测物体数: {vild_total_detections}")
    print(f"- 平均每图像检测物体数: {vild_avg_detections:.2f}")
    print(f"- 平均处理时间: {vild_avg_time:.3f}秒/图像")
    print(f"- 平均帧率: {vild_avg_fps:.2f} FPS")
    
    if has_rtdetr:
        print("\n原始RTDETR模型:")
        print(f"- 总检测物体数: {rtdetr_total_detections}")
        print(f"- 平均每图像检测物体数: {rtdetr_avg_detections:.2f}")
        print(f"- 平均处理时间: {rtdetr_avg_time:.3f}秒/图像")
        print(f"- 平均帧率: {rtdetr_avg_fps:.2f} FPS")
        
        # 计算性能提升
        detection_improvement = ((vild_avg_detections / rtdetr_avg_detections) - 1) * 100 if rtdetr_avg_detections > 0 else float('inf')
        speed_improvement = ((rtdetr_avg_time / vild_avg_time) - 1) * 100 if vild_avg_time > 0 else float('inf')
        
        print("\n性能提升:")
        print(f"- 检测能力提升: {vild_avg_detections - rtdetr_avg_detections:.2f}个物体/图像 ({detection_improvement:.1f}%)")
        
        if speed_improvement > 0:
            print(f"- 速度提升: {rtdetr_avg_time - vild_avg_time:.3f}秒/图像 ({speed_improvement:.1f}%)")
        else:
            print(f"- 速度变化: {vild_avg_time - rtdetr_avg_time:.3f}秒/图像 ({-speed_improvement:.1f}%)")
        
        # 打印类别差异
        vild_only = set(vild_category_counts.keys()) - set(rtdetr_category_counts.keys())
        rtdetr_only = set(rtdetr_category_counts.keys()) - set(vild_category_counts.keys())
        
        print("\n类别检测差异:")
        print(f"- 仅ViLD-RTDETR检测到的类别: {', '.join(vild_only) if vild_only else '无'}")
        print(f"- 仅原始RTDETR检测到的类别: {', '.join(rtdetr_only) if rtdetr_only else '无'}")
    
    print(f"\n✅ 详细比较结果已保存到: {output_dir}")
    return comparison_metrics

def main():
    """主函数"""
    # 直接在代码中指定图像路径/目录，不需要命令行参数
    
    # === 在这里设置评估参数 ===
    # 图像路径（单张图像或图像目录）
    image_path = "datasets/indoor_inference/images/indoor_000010.jpg"
    output_dir = os.path.join(project_root, "results/model_comparison")
    # 检测置信度阈值
    conf_threshold = 0.35
    # ========================
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 路径不存在: {image_path}")
        return
    
    print(f"🔍 开始比较模型性能: {image_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔢 置信度阈值: {conf_threshold}")
    
    try:
        compare_models(image_path, output_dir, conf_threshold)
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
