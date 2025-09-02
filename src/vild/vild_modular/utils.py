# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 工具函数
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import json
from collections import Counter
from scipy.ndimage import gaussian_filter

# 常用颜色映射
COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'brown': (165, 42, 42),
    'gray': (128, 128, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

def generate_colors(n):
    """生成n个不同的RGB颜色"""
    colors = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1
        s = 0.7 + 0.3 * ((i * 0.37) % 1)
        v = 0.7 + 0.3 * ((i * 0.73) % 1)
        
        # HSV到RGB转换
        r, g, b = hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors

def hsv_to_rgb(h, s, v):
    """HSV颜色转RGB"""
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if i % 6 == 0:
        r, g, b = v, t, p
    elif i % 6 == 1:
        r, g, b = q, v, p
    elif i % 6 == 2:
        r, g, b = p, v, t
    elif i % 6 == 3:
        r, g, b = p, q, v
    elif i % 6 == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    return r, g, b

def center_crop_resize(image, target_size):
    """中心裁剪并调整大小"""
    width, height = image.size
    
    # 计算中心裁剪区域
    square_size = min(width, height)
    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size
    
    # 裁剪并调整大小
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize(
        (target_size, target_size), 
        Image.Resampling.LANCZOS
    )
    
    return resized_image

def load_and_preprocess_image(image_path, target_size=224):
    """加载并预处理图像"""
    try:
        # 打开图像
        image = Image.open(image_path).convert('RGB')
        
        # 中心裁剪并调整大小
        processed_image = center_crop_resize(image, target_size)
        
        # 转换为张量并标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        tensor = transform(processed_image).unsqueeze(0)
        return tensor, image
    
    except Exception as e:
        print(f"图像处理错误 ({image_path}): {e}")
        return None, None

def draw_bbox(image, box, label="", score=None, color=(255, 0, 0), thickness=2):
    """在图像上绘制边界框和标签"""
    # 转换为PIL图像
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            # CHW转HWC
            image = image.permute(1, 2, 0)
        image_np = image.cpu().numpy()
        # 如果值范围在[0,1]内，转换到[0,255]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        image_pil = Image.fromarray(image_np)
    else:
        image_pil = image
    
    draw = ImageDraw.Draw(image_pil)
    
    # 获取边界框坐标
    x1, y1, x2, y2 = box
    
    # 绘制边界框
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)
    
    # 准备标签文本
    if score is not None:
        text = f"{label}: {score:.2f}"
    else:
        text = label
    
    # 计算文本背景框
    try:
        # 尝试加载Arial字体
        font_size = 12
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # 退回到默认字体
        font = ImageFont.load_default()
    
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    text_x = x1
    text_y = y1 - text_height - 2 if y1 > text_height + 2 else y1
    
    # 绘制标签背景和文本
    draw.rectangle(
        [(text_x, text_y), (text_x + text_width, text_y + text_height)],
        fill=color
    )
    draw.text(
        (text_x, text_y),
        text,
        fill=(255, 255, 255),
        font=font
    )
    
    return image_pil

def visualize_with_macro_categories(image, boxes, labels, scores=None, threshold=0.5, output_path=None):
    """使用大类别进行可视化检测结果"""
    try:
        # 导入大类别配置
        from config import MACRO_CATEGORIES
        
        # 将类别映射到大类别
        macro_labels = []
        for label in labels:
            found = False
            for macro_cat, items in MACRO_CATEGORIES.items():
                if label.lower() in [item.lower() for item in items]:
                    macro_labels.append(macro_cat)
                    found = True
                    break
            if not found:
                macro_labels.append(label)
                
        # 使用修改后的标签调用可视化函数
        return visualize_detections(image, boxes, macro_labels, scores, threshold, output_path)
    except ImportError:
        print("⚠️ 无法导入大类别配置，使用原始标签")
        return visualize_detections(image, boxes, labels, scores, threshold, output_path)

def visualize_detections(image, boxes, labels, scores=None, threshold=0.5, output_path=None):
    """可视化检测结果"""
    # 如果是张量或数组，转换为PIL图像
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            # CHW转HWC
            image = image.permute(1, 2, 0)
        image_np = image.cpu().numpy()
        # 如果值范围在[0,1]内，转换到[0,255]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        image = Image.fromarray(image_np)
    
    # 复制图像，以免修改原始图像
    output_image = image.copy()
    
    # 确保boxes和labels是列表
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().tolist()
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy().tolist()
    
    # 获取唯一标签和颜色映射
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    
    colors = generate_colors(len(unique_labels))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 绘制每个检测框
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # 如果有分数，检查阈值
        if scores is not None:
            score = scores[i]
            if score < threshold:
                continue
        else:
            score = None
        
        # 获取颜色
        color = color_map.get(label, (255, 0, 0))
        
        # 绘制边界框和标签
        output_image = draw_bbox(
            output_image, box, label, score, color, thickness=2
        )
    
    # 如果指定了输出路径，保存图像
    if output_path:
        output_image.save(output_path)
    
    return output_image

def calculate_iou(box1, box2):
    """计算两个边界框之间的IoU"""
    # 提取坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域坐标
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # 检查交集是否存在
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算各自面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou

def non_max_suppression(boxes, scores, labels, iou_threshold=0.5):
    """非极大值抑制"""
    # 若没有检测框，直接返回空列表
    if len(boxes) == 0:
        return [], [], []
    
    # 确保输入是NumPy数组
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 按分数降序排列
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    
    keep = []
    
    while len(boxes) > 0:
        # 保留分数最高的边界框
        keep.append(0)
        
        # 如果只剩一个边界框，就结束循环
        if len(boxes) == 1:
            break
        
        # 计算其余边界框与当前最高分数边界框的IoU
        ious = np.array([calculate_iou(boxes[0], box) for box in boxes[1:]])
        
        # 找出低于阈值的框的索引
        valid_indices = np.where(ious < iou_threshold)[0]
        
        # 更新boxes, scores和labels，只保留有效的框
        boxes = np.vstack([boxes[0:1], boxes[valid_indices + 1]])
        scores = np.concatenate([scores[0:1], scores[valid_indices + 1]])
        labels = np.concatenate([labels[0:1], labels[valid_indices + 1]])
    
    # 收集结果
    keep_boxes = [boxes[i] for i in keep]
    keep_scores = [scores[i] for i in keep]
    keep_labels = [labels[i] for i in keep]
    
    return np.array(keep_boxes), np.array(keep_scores), np.array(keep_labels)

def group_boxes_by_image(detections):
    """将检测结果按图像分组"""
    grouped_detections = {}
    
    for detection in detections:
        image_path = detection['image_path']
        if image_path not in grouped_detections:
            grouped_detections[image_path] = []
        
        grouped_detections[image_path].append(detection)
    
    return grouped_detections

def save_detection_results(detections, output_file):
    """保存检测结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    
    print(f"检测结果已保存到 {output_file}")

def calculate_detection_stats(detections):
    """计算检测统计信息"""
    # 统计所有检测的类别
    all_categories = []
    for detection in detections:
        all_categories.extend(detection['categories'])
    
    # 计算类别频率
    category_counts = Counter(all_categories)
    
    # 计算总检测框数
    total_boxes = sum(category_counts.values())
    
    # 计算每个类别的百分比
    category_percentages = {
        category: (count / total_boxes) * 100 
        for category, count in category_counts.items()
    }
    
    # 统计每张图像的检测框数量
    images_with_detections = len(set(d['image_path'] for d in detections))
    avg_boxes_per_image = total_boxes / max(1, images_with_detections)
    
    # 计算置信度分布
    all_scores = []
    for detection in detections:
        if 'scores' in detection:
            all_scores.extend(detection['scores'])
    
    score_stats = {}
    if all_scores:
        score_stats = {
            'min': float(min(all_scores)),
            'max': float(max(all_scores)),
            'mean': float(np.mean(all_scores)),
            'median': float(np.median(all_scores)),
            'std': float(np.std(all_scores))
        }
    
    stats = {
        'total_detections': total_boxes,
        'total_images': images_with_detections,
        'avg_boxes_per_image': avg_boxes_per_image,
        'category_counts': category_counts,
        'category_percentages': category_percentages,
        'score_stats': score_stats
    }
    
    return stats

def timer(func):
    """函数计时装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 运行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

def enhance_image_for_detection(image_path, output_path=None):
    """
    增强图像以改善检测效果
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径，如果为None则返回增强后的PIL图像
        
    返回:
        如果output_path为None，返回增强后的PIL图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为RGB（OpenCV默认为BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建增强图像副本
    enhanced = image_rgb.copy()
    
    # 1. 应用自适应直方图均衡化来提高对比度
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 2. 适度锐化以增强边缘
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpening)
    
    # 3. 适度的去噪
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    
    # 转换为PIL图像
    enhanced_pil = Image.fromarray(enhanced)
    
    # 如果指定了输出路径，保存图像
    if output_path:
        enhanced_pil.save(output_path)
        return None
        
    return enhanced_pil
