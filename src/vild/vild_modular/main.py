# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 主程序
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import cv2
import traceback
import time
from PIL import Image
        
import clip
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from model import load_models
from data_loader import load_coco_indoor, select_random_test_image
from detector import FixedViLDDetector
from training import run_fixed_training
from utils import visualize_detections, save_detection_results, calculate_detection_stats, visualize_with_macro_categories
from config import MODEL_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG

# 全局控制变量
ENABLE_TRAINING = False  # 控制是否执行训练过程
ENABLE_DETECTION = True  # 控制是否执行检测过程
ENABLE_LOAD_MODEL = True  # 控制是否加载训练好的模型
# 直接指定测试图像的路径，如果设置了具体路径，将优先使用此路径
TEST_IMAGE_PATH = None # 指定测试图像的路径
TEST_IMAGE_INDEX = -1     # 指定数据集中的图像索引，0表示使用第一张图像（仅在TEST_IMAGE_PATH为None时生效）

# 可选的命令行参数解析
def select_random_test_image(images, image_root, index=None):
    """从数据集中选择一个测试图像"""
    if not images:
        raise ValueError("没有可用的图像")
    
    if index is not None and 0 <= index < len(images):
        image_info = images[index]
    else:
        image_info = random.choice(images)
    
    image_path = os.path.join(image_root, image_info['file_name'])
    return image_path

def parse_args():
    """解析命令行参数（仅用于覆盖全局配置）"""
    global ENABLE_TRAINING, ENABLE_DETECTION, ENABLE_LOAD_MODEL, TEST_IMAGE_PATH, TEST_IMAGE_INDEX
    
    parser = argparse.ArgumentParser(description='ViLD室内检测')
    parser.add_argument('--train', action='store_true', help='启用训练模式')
    parser.add_argument('--detect', action='store_true', help='启用检测模式')
    parser.add_argument('--load-model', action='store_true', help='加载训练好的模型')
    parser.add_argument('--model-path', type=str, help='指定要加载的模型路径')
    parser.add_argument('--image', type=str, help='检测使用的图像路径')
    parser.add_argument('--image-index', type=int, help='使用数据集中指定索引的图像进行检测')
    parser.add_argument('--output-dir', type=str, default=None, help='检测结果输出目录')
    
    args = parser.parse_args()
    
    # 使用命令行参数覆盖全局配置
    if args.train:
        ENABLE_TRAINING = True
    if args.detect:
        ENABLE_DETECTION = True
    if args.load_model:
        ENABLE_LOAD_MODEL = True
    if args.image:
        TEST_IMAGE_PATH = args.image
    if args.image_index is not None:
        TEST_IMAGE_INDEX = args.image_index
    
    # 如果指定了模型路径，修改配置
    if args.model_path:
        MODEL_CONFIG['projector_path'] = args.model_path
    
    return args

def run_training(clip_model, device):
    """运行训练过程"""
    if not ENABLE_TRAINING:
        print("⏭️ 训练功能已禁用，跳过训练过程")
        return
    
    print("� 开始优化版ViLD训练")
    print("=" * 100)
    
    try:
        # 配置数据路径
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # 使用筛选后的室内场景数据集进行训练
        COCO_PATH = os.path.join(PROJECT_ROOT, "datasets/indoor_enhanced/coco_indoor_train.json")
        IMAGE_ROOT = os.path.join(PROJECT_ROOT, "datasets/coco/train2017")  # 原始图像路径
        
        # 检查数据文件是否存在
        if not os.path.exists(COCO_PATH):
            print(f"❌ 训练数据文件不存在: {COCO_PATH}")
            print(f"尝试使用备用数据集...")
            # 备选训练数据
            COCO_PATH = os.path.join(PROJECT_ROOT, "datasets/indoor_training/annotations_train.json")
            IMAGE_ROOT = os.path.join(PROJECT_ROOT, "datasets/indoor_training/train")
        
        print(f"📊 正在加载训练数据: {COCO_PATH}")
        print(f"📸 图像根目录: {IMAGE_ROOT}")
        
        images, categories = load_coco_indoor(COCO_PATH, IMAGE_ROOT)
        
        if not images:
            print("❌ 无法加载训练数据")
            return
            
        print(f"✅ 成功加载 {len(images)} 张训练图像")
        
        # 运行训练
        run_fixed_training(
            clip_model=clip_model,
            device=device,
            images=images,
            image_root=IMAGE_ROOT
        )
    
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        traceback.print_exc()

def run_detection(clip_model, rtdetr_model, image_processor, clip_preprocess, device, args=None):
    """运行检测过程"""
    if not ENABLE_DETECTION:
        print("⏭️ 检测功能已禁用，跳过检测过程")
        return
    
    # 设置输出目录
    output_dir = None
    if args and args.output_dir:
        output_dir = args.output_dir
        
    # 设置更高的检测阈值，特别是对于用户提供的图像
    custom_threshold = 0.45  # 使用更高的阈值减少误检测
        
    print("🔄 开始物体检测...")
    
    try:
        # 创建场景分类器
        from scene_classifier import SceneClassifier
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
        
        # 配置项目路径
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        VILD_DIR = os.path.dirname(os.path.dirname(__file__))  # vild目录
        
        # 设置输出目录在vild文件夹下
        if output_dir is None:
            output_dir = os.path.join(VILD_DIR, "results")
        else:
            output_dir = os.path.join(VILD_DIR, output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 结果将保存在: {output_dir}")
        
        # 获取测试图像
        test_image_path = TEST_IMAGE_PATH
        if not test_image_path:
            # 如果没有指定图像路径，尝试从筛选的室内数据集随机选择
            COCO_PATH = os.path.join(PROJECT_ROOT, "datasets/indoor_enhanced/coco_indoor_val.json")
            IMAGE_ROOT = os.path.join(PROJECT_ROOT, "datasets/coco/train2017")
            
            # 检查筛选后的验证数据集是否存在
            if not os.path.exists(COCO_PATH):
                print(f"⚠️ 验证数据集不存在: {COCO_PATH}")
                # 尝试筛选后的多个场景数据集
                scene_datasets = [
                    os.path.join(PROJECT_ROOT, "datasets/indoor_scenes/coco_bathroom_subset.json"),
                    os.path.join(PROJECT_ROOT, "datasets/indoor_scenes/coco_kitchen_subset.json"),
                    os.path.join(PROJECT_ROOT, "datasets/indoor_scenes/coco_bedroom_subset.json"),
                    os.path.join(PROJECT_ROOT, "datasets/indoor_scenes/coco_living_room_subset.json")
                ]
                
                for scene_dataset in scene_datasets:
                    if os.path.exists(scene_dataset):
                        print(f"✅ 找到场景数据集: {scene_dataset}")
                        COCO_PATH = scene_dataset
                        break
                else:
                    # 如果筛选的数据集不存在，尝试使用原始数据集
                    COCO_PATH = os.path.join(PROJECT_ROOT, "datasets/indoor_training/annotations_train.json")
                    IMAGE_ROOT = os.path.join(PROJECT_ROOT, "datasets/indoor_training/train")
            
            try:
                images, _ = load_coco_indoor(COCO_PATH, IMAGE_ROOT)
                test_image_path = select_random_test_image(images, IMAGE_ROOT, TEST_IMAGE_INDEX)
            except Exception as e:
                print(f"⚠️ 无法从数据集选择图像: {e}")
        
        # 如果仍然没有图像，创建测试图像
        if not test_image_path or not os.path.exists(test_image_path):
            print("创建测试图像...")
            test_dir = os.path.join(PROJECT_ROOT, "tests")
            os.makedirs(test_dir, exist_ok=True)
            test_image_path = os.path.join(test_dir, "test_image.jpg")
            
            # 创建简单的测试图像
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
            cv2.rectangle(test_image, (100, 100), (300, 300), (0, 0, 255), 2)
            cv2.circle(test_image, (400, 200), 50, (0, 255, 0), -1)
            cv2.imwrite(test_image_path, test_image)
            print(f"✓ 已创建测试图像: {test_image_path}")
        
        print(f"📷 使用测试图像: {test_image_path}")
        
        # 图像预处理增强 - 提高检测质量
        from utils import enhance_image_for_detection
        try:
            print("🔄 应用图像增强预处理...")
            enhanced_image = enhance_image_for_detection(test_image_path)
            print("✓ 图像预处理完成")
        except Exception as e:
            print(f"⚠️ 图像增强失败: {e}，使用原始图像")
            enhanced_image = None
        
        # 先进行场景分类
        print("\n🔍 开始场景分类...")
        if enhanced_image:
            scene_type, scene_score = scene_classifier.classify_scene(enhanced_image)
        else:
            scene_type, scene_score = scene_classifier.classify_scene(test_image_path)
            
        scene_context = scene_classifier.get_scene_type(scene_type)
        print(f"🏠 识别场景: {scene_type} (场景类型: {scene_context}, 置信度: {scene_score:.3f})")
        
        # 根据场景类型运行检测，但不设置特殊优先级
        print("\n🔍 开始物体检测...")
        # 使用大类别检测模式
        use_macro_categories = True  # 默认启用大类别模式
        
        # 简化场景类型处理，不设优先级
        print(f"� 使用识别的场景类型: {scene_type}")
        
        # 直接使用场景分类器识别的场景类型
        detect_image = enhanced_image if enhanced_image else test_image_path
        result = detector.detect_objects(detect_image, scene_type=scene_type, use_macro_categories=use_macro_categories)
        
        # 保存检测结果
        if result:
            # 获取检测信息
            boxes = result['boxes']
            scores = result['scores']
            categories = result['labels']
            
            # 将检测结果保存到文件
            detection_info = {
                'image_path': test_image_path,
                'boxes': boxes.tolist() if isinstance(boxes, (np.ndarray, torch.Tensor)) else boxes,
                'categories': categories,
                'scores': scores.tolist() if isinstance(scores, (np.ndarray, torch.Tensor)) else scores
            }
            
            # 保存结果
            results_file = os.path.join(output_dir, 'detection_result.json')
            save_detection_results([detection_info], results_file)
            
            # 可视化结果
            image = Image.open(test_image_path).convert('RGB')
            # 使用大类别可视化
            vis_image = visualize_with_macro_categories(
                image, boxes, categories, scores, 
                threshold=INFERENCE_CONFIG['score_threshold']
            )
            
            # 保存可视化结果
            output_basename = f"detection_result_{os.path.basename(test_image_path)}"
            output_path = os.path.join(output_dir, output_basename)
            vis_image.save(output_path)
            print(f"✓ 已保存检测结果: {output_path}")
            
            # 同时保存包含场景类型的结果
            scene_output_path = os.path.join(output_dir, f"scene_{scene_type}_{os.path.basename(test_image_path)}")
            vis_image.save(scene_output_path)
            print(f"✓ 已保存场景检测结果: {scene_output_path}")
            
            # 打印检测统计信息
            if len(boxes) > 0:
                print(f"\n📊 检测结果: 找到 {len(boxes)} 个物体")
                for i, (label, score) in enumerate(zip(categories, scores)):
                    print(f"  {i+1}. {label}: {score:.2f}")
            else:
                print("⚠️ 未检测到物体")
        else:
            print("❌ 检测失败")
    
    except Exception as e:
        print(f"❌ 检测出错: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    start_time = time.time()
    
    # 可以解析命令行参数来覆盖全局配置
    args = parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 打印模型路径信息
    if ENABLE_LOAD_MODEL:
        print(f"📂 使用训练模型路径: {MODEL_CONFIG['projector_path']}")
    
    # 加载模型
    print("📦 正在加载模型...")
    rtdetr_model, clip_model, image_processor, clip_preprocess = load_models(
        rtdetr_path=MODEL_CONFIG['rtdetr_model_path'],
        clip_name=MODEL_CONFIG['clip_model_name'],
        device=device
    )
    
    # 打印当前配置
    print(f"\n⚙️ 运行配置:")
    print(f"   训练模式: {'启用' if ENABLE_TRAINING else '禁用'}")
    print(f"   检测模式: {'启用' if ENABLE_DETECTION else '禁用'}")
    print(f"   加载模型: {'启用' if ENABLE_LOAD_MODEL else '禁用'}")
    print(f"   模型路径: {MODEL_CONFIG['projector_path']}")
    print(f"   测试图像: {TEST_IMAGE_PATH if TEST_IMAGE_PATH else '自动选择'}")
    print(f"   测试索引: {TEST_IMAGE_INDEX}")
    
    # 运行训练
    run_training(clip_model, device)
    
    # 运行检测
    run_detection(clip_model, rtdetr_model, image_processor, clip_preprocess, device, args)
    
    # 打印总运行时间
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 总运行时间: {elapsed_time:.2f}秒")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        import traceback
        print(f"程序出错: {e}")
        traceback.print_exc()
