#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ViLD-RTDETR 模型性能评估脚本
此脚本用于评估模型在室内场景检测任务上的性能
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加项目根目录到模块搜索路径
sys.path.append(project_root)

def evaluate_model(image_path, output_dir=None, conf_threshold=0.35):
    """评估模型在单张图像或图像目录上的性能"""
    from src.vild.vild_modular.config import MODEL_CONFIG, INFERENCE_CONFIG
    from src.vild.vild_modular.detector import FixedViLDDetector
    from src.vild.vild_modular.scene_classifier import SceneClassifier
    from src.vild.vild_modular.model import load_models
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(current_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 结果将保存在: {output_dir}")
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 加载模型
    print("📦 正在加载模型...")
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
    
    # 创建检测器
    detector = FixedViLDDetector(
        clip_model=clip_model,
        detector_model=rtdetr_model,
        image_processor=image_processor,
        clip_preprocess=clip_preprocess,
        device=device,
        projector_path=MODEL_CONFIG['projector_path'],
        config={'model': MODEL_CONFIG, 'inference': INFERENCE_CONFIG}
    )
    
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
    results = []
    total_time = 0
    category_counts = {}
    
    # 处理每个图像
    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 记录开始时间
            start_time = time.time()
            
            # 场景分类
            scene_type, scene_score = scene_classifier.classify_scene(image)
            
            # 物体检测
            detection_result = detector.detect_objects(
                image, 
                scene_type=scene_type,
                use_macro_categories=True  # 使用大类别分组
            )
            
            # 记录结束时间
            end_time = time.time()
            process_time = end_time - start_time
            total_time += process_time
            
            # 获取检测结果
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
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                # 可视化结果
                draw = ImageDraw.Draw(image)
                
                # 尝试加载字体
                try:
                    font = ImageFont.truetype("Arial.ttf", 15)
                except:
                    font = ImageFont.load_default()
                
                # 绘制检测框
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
                
                # 在图像顶部添加场景信息
                scene_info = f"场景: {scene_type} ({scene_score:.2f}) - 处理时间: {process_time:.3f}s"
                draw.rectangle([0, 0, len(scene_info) * 8, 20], fill="black")
                draw.text((5, 5), scene_info, fill="white", font=font)
                
                # 保存结果图像
                output_file = os.path.join(output_dir, os.path.basename(img_path))
                image.save(output_file)
                
                # 收集结果信息
                results.append({
                    'image_path': img_path,
                    'scene_type': scene_type,
                    'scene_score': float(scene_score),
                    'process_time': process_time,
                    'detections': [
                        {
                            'bbox': box.tolist() if isinstance(box, np.ndarray) else box,
                            'score': float(score),
                            'category': cat
                        }
                        for box, score, cat in zip(valid_boxes, valid_scores, valid_categories)
                    ]
                })
                
                print(f"✓ 处理图像 {os.path.basename(img_path)}: 场景={scene_type}, "
                      f"检测到 {len(valid_boxes)} 个物体, 用时 {process_time:.3f}s")
            else:
                print(f"⚠️ 图像 {os.path.basename(img_path)} 未检测到物体")
        
        except Exception as e:
            print(f"❌ 处理图像 {os.path.basename(img_path)} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算总体统计信息
    avg_time = total_time / len(image_files) if image_files else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    total_detections = sum(len(r['detections']) for r in results)
    avg_detections = total_detections / len(results) if results else 0
    
    # 生成统计报告
    stats = {
        'total_images': len(image_files),
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'total_processing_time': total_time,
        'avg_processing_time': avg_time,
        'avg_fps': avg_fps,
        'category_statistics': {
            cat: count for cat, count in sorted(
                category_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        }
    }
    
    # 保存详细结果
    results_file = os.path.join(output_dir, 'detection_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'detection_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 绘制类别分布饼图
    if category_counts:
        plt.figure(figsize=(10, 8))
        
        # 限制显示前10个类别，其余归为"其他"
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        if len(top_categories) > 10:
            top_10 = top_categories[:10]
            others_count = sum(count for _, count in top_categories[10:])
            chart_data = dict(top_10)
            chart_data['其他'] = others_count
        else:
            chart_data = dict(top_categories)
        
        # 绘制饼图
        plt.pie(
            chart_data.values(), 
            labels=chart_data.keys(), 
            autopct='%1.1f%%',
            shadow=True, 
            startangle=140
        )
        plt.axis('equal')
        plt.title('检测物体类别分布')
        
        # 保存图表
        chart_file = os.path.join(output_dir, 'category_distribution.png')
        plt.savefig(chart_file)
    
    # 打印统计信息
    print("\n===== 评估结果摘要 =====")
    print(f"总图像数: {stats['total_images']}")
    print(f"总检测物体数: {stats['total_detections']}")
    print(f"平均每图像检测物体数: {stats['avg_detections_per_image']:.2f}")
    print(f"总处理时间: {stats['total_processing_time']:.2f}秒")
    print(f"平均处理时间: {stats['avg_processing_time']:.3f}秒/图像")
    print(f"平均帧率: {stats['avg_fps']:.2f} FPS")
    
    print("\n物体类别统计 (前10):")
    for i, (cat, count) in enumerate(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
        print(f"{i+1}. {cat}: {count} 个")
    
    print(f"\n✅ 详细结果已保存到: {output_dir}")
    return stats

def main():
    """主函数"""
    # 直接在代码中指定图像路径/目录，不需要命令行参数
    
    # === 在这里设置评估参数 ===
    # 图像路径（单张图像或图像目录）
    image_path = "datasets/indoor_inference/images"
    # 输出目录路径
    output_dir = "results/model_evaluation"
    # 检测置信度阈值
    conf_threshold = 0.35
    # ========================
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 路径不存在: {image_path}")
        return
    
    print(f"🔍 开始评估图像: {image_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔢 置信度阈值: {conf_threshold}")
    
    try:
        evaluate_model(image_path, output_dir, conf_threshold)
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
