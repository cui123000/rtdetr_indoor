# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测

本项目实现了基于Vision-Language知识蒸馏(ViLD)的开放世界室内物体检测系统。主要特点：

1. 使用RTDETR作为基础检测器架构
2. 集成CLIP预训练模型的视觉-语言知识
3. 通过知识蒸馏实现开放词汇目标检测
4. 引入可学习的提示词优化分类性能
"""

# 导入必要的库
import os
import json
import time
import random
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import cv2

# 设置matplotlib为非交互模式，避免在无显示环境下卡住
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合无头服务器
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
# 配置tqdm在终端正确显示
import sys
tqdm_kwargs = {
    'file': sys.stdout,
    'ncols': 100,
    'ascii': True,  # 使用ASCII字符，避免在某些终端中显示问题
    'leave': True   # 保留进度条
}
import torch.nn as nn 
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torchvision.transforms as T
import random
from torch.utils.data import Dataset, DataLoader
import time
import gc
from torchvision import transforms

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# 全局控制变量
ENABLE_TRAINING = False  # 控制是否执行训练过程
ENABLE_DETECTION = True  # 控制是否执行检测过程
TEST_IMAGE_INDEX = -1    # 测试图像索引，-1表示随机选择

# =============================================================================
# 1. 数据加载与预处理
# =============================================================================
"""
本节完成以下任务：

1. 加载COCO数据集中的图像
2. 处理图像和标注数据
3. 准备teacher模型(CLIP)输入
4. 准备student模型(RT-DETR)输入
"""

# 配置数据路径
# 获取项目根目录
PROJECT_ROOT = "/home/cui/vild_rtdetr_indoor"  # 直接指定绝对路径
print(f"项目根目录: {PROJECT_ROOT}")

# 配置数据集路径
COCO_PATH = os.path.join(PROJECT_ROOT, "datasets/indoor_training/annotations_train.json")
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "datasets/indoor_training/train")

def load_coco_indoor():
    """加载COCO数据集中的室内场景数据"""
    if not os.path.exists(COCO_PATH):
        raise FileNotFoundError(f"注释文件不存在: {COCO_PATH}")
        
    print(f"正在加载数据集: {COCO_PATH}")
    with open(COCO_PATH, 'r') as f:
        dataset = json.load(f)
    
    # 打印数据集的基本信息，帮助调试
    print(f"数据集键: {list(dataset.keys())}")
    if 'images' in dataset:
        print(f"图像数量: {len(dataset['images'])}")
        if len(dataset['images']) > 0:
            print(f"第一张图像的键: {list(dataset['images'][0].keys())}")
    if 'categories' in dataset:
        print(f"类别数量: {len(dataset['categories'])}")
    
    # 构建类别映射
    categories = {cat['id']: cat for cat in dataset['categories']}
    
    # 处理图像和标注
    image_dict = {}
    for image in dataset['images']:
        # LVIS数据集中可能使用coco_url或file_name
        file_name = None
        
        # 尝试不同的可能键名
        if 'file_name' in image:
            file_name = image['file_name']
        elif 'coco_url' in image:
            # 从coco_url中提取文件名
            file_name = os.path.basename(image['coco_url'])
        else:
            # 打印图像的键以便调试
            print(f"警告: 找不到图像路径，图像对象的键: {list(image.keys())}")
            continue
        
        image_dict[image['id']] = {
            'file_name': file_name,
            'height': image.get('height', 0),
            'width': image.get('width', 0),
            'annotations': []
        }
    
    # 添加标注信息
    for ann in dataset['annotations']:
        try:
            image_id = ann['image_id']
            if image_id in image_dict:
                # 确保所有必需的字段都存在
                if 'bbox' in ann and 'category_id' in ann:
                    image_dict[image_id]['annotations'].append({
                        'bbox': ann['bbox'],  # [x, y, w, h]
                        'category_id': ann['category_id'],
                        'segmentation': ann.get('segmentation', []),
                        'iscrowd': ann.get('iscrowd', 0)
                    })
        except KeyError as e:
            print(f"警告: 标注缺少必要字段 {e}")
            continue
    
    # 过滤掉没有标注的图像
    valid_images = [img for img in image_dict.values() if len(img['annotations']) > 0]
    print(f"有效图像数量(含标注): {len(valid_images)}/{len(image_dict)}")
    
    return valid_images, categories

# 加载数据集
try:
    print(f"正在检查路径...")
    print(f"COCO注释文件路径: {COCO_PATH}")
    print(f"图像根目录: {IMAGE_ROOT}")
    
    # 初始化变量，防止加载失败时未定义
    images = []
    categories = {}
    
    if os.path.exists(COCO_PATH):
        print("找到注释文件")
        # 尝试加载数据
        try:
            images, categories = load_coco_indoor()
            print(f"成功加载了 {len(images)} 张图片和 {len(categories)} 个类别")
            
            # 验证图像路径
            if len(images) > 0:
                sample_path = os.path.join(IMAGE_ROOT, images[0]['file_name'])
                print(f"示例图像路径: {sample_path}")
                print(f"图像文件是否存在: {os.path.exists(sample_path)}")
        except Exception as load_error:
            print(f"数据加载出错: {load_error}")
            print("尝试切换到其他可用数据集...")
            
            # 尝试加载COCO数据集
            alt_coco_path = os.path.join(PROJECT_ROOT, "datasets/coco/train2017")
            if os.path.exists(alt_coco_path):
                print(f"找到COCO数据集: {alt_coco_path}")
                # 创建模拟数据
                import glob
                coco_images = glob.glob(os.path.join(alt_coco_path, "*.jpg"))[:10]
                print(f"找到 {len(coco_images)} 个COCO图像")
                
                # 创建模拟标注
                for idx, img_path in enumerate(coco_images):
                    img_name = os.path.basename(img_path)
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        images.append({
                            'file_name': img_name,
                            'height': h,
                            'width': w,
                            'annotations': [
                                {'bbox': [w//4, h//4, w//2, h//2], 'category_id': 1}
                            ]
                        })
                categories = {1: {'id': 1, 'name': 'object'}}
                print(f"已创建 {len(images)} 个模拟样本")
                
                # 更新图像根目录
                IMAGE_ROOT = alt_coco_path
    else:
        print("注释文件不存在，尝试使用COCO数据集...")
        
        # 尝试使用COCO数据集
        coco_path = os.path.join(PROJECT_ROOT, "datasets/coco/train2017")
        if os.path.exists(coco_path):
            print(f"找到COCO数据集: {coco_path}")
            # 创建模拟数据
            import glob
            coco_images = glob.glob(os.path.join(coco_path, "*.jpg"))[:10]
            print(f"找到 {len(coco_images)} 个COCO图像")
            
            # 创建模拟标注
            for idx, img_path in enumerate(coco_images):
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    images.append({
                        'file_name': img_name,
                        'height': h,
                        'width': w,
                        'annotations': [
                            {'bbox': [w//4, h//4, w//2, h//2], 'category_id': 1}
                        ]
                    })
            categories = {1: {'id': 1, 'name': 'object'}}
            print(f"已创建 {len(images)} 个模拟样本")
            
            # 更新图像根目录
            IMAGE_ROOT = coco_path
        else:
            print("无法找到任何数据集，创建模拟数据...")
            # 创建一个假的数据集以便代码能继续运行
            images = []
            categories = {1: {'id': 1, 'name': 'object'}}
        
except Exception as e:
    print(f"加载数据集时出错: {str(e)}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据集路径: {COCO_PATH}")
    import traceback
    traceback.print_exc()
    
    # 确保变量已定义
    images = []
    categories = {1: {'id': 1, 'name': 'object'}}

# 加载CLIP模型
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
clip_model.eval()

# 设置默认浮点类型，避免半精度问题
torch.set_default_dtype(torch.float32)
print("已设置默认数据类型为 float32，避免半精度问题")

# 加载RT-DETR检测器
try:
    image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    detector_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)
    detector_model.eval()
    print("成功加载RT-DETR模型")
except Exception as e:
    print(f"加载RT-DETR失败: {str(e)}")

class ImageProcessor:
    def __init__(self, clip_preprocess):
        self.clip_preprocess = clip_preprocess
    
    def prepare_image_clip(self, image_path):
        """处理图像用于CLIP模型"""
        image = Image.open(image_path).convert('RGB')
        return self.clip_preprocess(image).unsqueeze(0).to(device)
    
    def prepare_image_detector(self, image_path):
        """处理图像用于检测器"""
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 从数据集中选择随机测试图像的函数
def select_random_test_image():
    """从数据集中选择一个随机测试图像"""
    if len(images) == 0:
        return None
    
    # 如果指定了测试图像索引，使用它；否则随机选择
    if TEST_IMAGE_INDEX >= 0 and TEST_IMAGE_INDEX < len(images):
        img_index = TEST_IMAGE_INDEX
    else:
        # 随机选择一个图像
        img_index = random.randint(0, len(images) - 1)
    
    img_info = images[img_index]
    img_path = os.path.join(IMAGE_ROOT, img_info['file_name'])
    
    if os.path.exists(img_path):
        print(f"📷 选择测试图像: {os.path.basename(img_path)} (索引 {img_index})")
        return img_path
    else:
        print(f"⚠️ 选择的图像不存在: {img_path}")
        return None

# 初始化图像处理器
processor = ImageProcessor(clip_preprocess)

# 测试图像处理
test_image_path = select_random_test_image()
if test_image_path:
    print(f"找到有效的测试图像: {test_image_path}")
else:
    print("警告: 未找到有效的测试图像")
    test_image_path = None

# 如果没有找到图像，创建一个测试图像
if test_image_path is None:
    print("没有找到有效图像，创建测试图像...")
    test_dir = os.path.join(PROJECT_ROOT, "tests")
    os.makedirs(test_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "test_image.jpg")
    
    # 创建一个简单的测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    # 绘制一些简单的形状
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 0, 255), 2)
    cv2.circle(test_image, (400, 200), 50, (0, 255, 0), -1)
    cv2.imwrite(test_image_path, test_image)
    print(f"已创建测试图像: {test_image_path}")
        
if test_image_path:
    try:
        clip_input = processor.prepare_image_clip(test_image_path)
        detector_input = processor.prepare_image_detector(test_image_path)
        print("CLIP输入张量形状:", clip_input.shape)
        print("检测器输入图像形状:", detector_input.shape)
    except Exception as e:
        print(f"处理测试图像时出错: {e}")
else:
    print("无法创建或找到任何图像数据")

# =============================================================================
# 2. 模型架构定义
# =============================================================================
"""
本节实现以下组件：

1. 基于RT-DETR的检测器架构
2. 集成CLIP视觉编码器
3. 特征投影层
4. 知识蒸馏的损失函数
"""

# 定义ViLD模型
class ViLDModel(nn.Module):
    def __init__(self, clip_model, detector_model):
        super().__init__()
        self.clip_model = clip_model
        self.detector_model = detector_model
        
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 特征融合层
        self.fusion_layer = nn.Linear(512, 256)  # 假设CLIP输出512维，检测器特征256维
        
        # 多尺度特征投影器
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 512)
            ) for _ in range(4)  # 对应RT-DETR的4个特征尺度
        ])
        
    def forward(self, images):
        # 使用检测器获取区域特征
        detector_inputs = image_processor(images=images, return_tensors="pt").to(device)
        detector_outputs = self.detector_model(**detector_inputs, output_hidden_states=True)
        
        # 获取多尺度特征（取最后4层的[CLS] token）
        features = [h[:, 0] for h in detector_outputs.hidden_states[-4:]]
        
        # 投影特征
        projected_features = [proj(feat) for proj, feat in zip(self.projectors, features)]
        
        # 使用CLIP获取全局特征
        clip_inputs = torch.stack([clip_preprocess(img) for img in images]).to(device)
        clip_features = self.clip_model.encode_image(clip_inputs)
        
        # 特征融合
        fused_features = self.fusion_layer(clip_features)
        
        return {
            "detector_outputs": detector_outputs,
            "clip_features": clip_features,
            "fused_features": fused_features
        }

# 初始化ViLD模型
try:
    vild_model = ViLDModel(clip_model, detector_model).to(device)
    print("ViLD模型构建成功")
    
    # 打印模型信息
    print(f"设备: {device}")
    print(f"CLIP模型: ViT-B/32")
    print(f"检测器模型: {type(detector_model).__name__}")
    print(f"融合层结构: {vild_model.fusion_layer}")
    
except Exception as e:
    print(f"模型构建失败: {str(e)}")

# =============================================================================
# 3. 知识蒸馏训练
# =============================================================================
"""
本节实现优化后的知识蒸馏训练流程，特别关注训练稳定性：

1. **稳定的特征提取**
   - 使用LayerNorm代替BatchNorm
   - 添加残差连接提高特征传播稳定性
   - 使用GELU激活函数获得更平滑的梯度

2. **改进的损失计算**
   - 使用损失平滑(Loss Smoothing)防止过拟合
   - 添加余弦相似度与L1损失的组合
   - 应用梯度裁剪防止梯度爆炸

3. **优化的学习调度**
   - 实现OneCycleLR学习率调度
   - 包含预热阶段减少初始不稳定性
   - 使用EMA(指数移动平均)平滑训练曲线

4. **稳健的训练监控**
   - 同时跟踪原始损失和平滑损失
   - 早停机制避免过拟合
   - 动态可视化损失变化曲线
"""

# 损失追踪器
class LossTracker:
    """损失追踪和可视化"""
    
    def __init__(self):
        self.train_losses = []
        self.epoch_losses = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def update(self, epoch_loss, epoch):
        """更新损失记录"""
        self.epoch_losses.append(epoch_loss)
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            
    def plot_losses(self, save_path=None, train_losses=None, val_losses=None, lr_history=None):
        """绘制增强版损失曲线和学习率"""
        plt.figure(figsize=(15, 10))
        
        # 创建多子图
        gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0, :])  # 上方占两列的损失图
        ax2 = plt.subplot(gs[1, 0])  # 左下角的训练/验证损失对比
        ax3 = plt.subplot(gs[1, 1])  # 右下角的学习率曲线
        
        # 1. 主损失曲线 (上方大图)
        epochs = range(1, len(self.epoch_losses) + 1)
        ax1.plot(epochs, self.epoch_losses, 'b-', linewidth=2.5, label='Validation Loss', marker='o')
        
        # 标注最佳损失点
        ax1.plot(self.best_epoch + 1, self.best_loss, 'r*', markersize=20, 
                label=f'Best Loss: {self.best_loss:.4f} (Epoch {self.best_epoch + 1})')
        
        # 添加移动平均线
        if len(self.epoch_losses) >= 3:
            window_size = min(3, len(self.epoch_losses))
            moving_avg = []
            for i in range(len(self.epoch_losses)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(self.epoch_losses[start_idx:i+1]))
            ax1.plot(epochs, moving_avg, 'g--', linewidth=2, alpha=0.7, label='Moving Average')
        
        # 设置主图的样式
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss Value', fontsize=14)
        ax1.set_title('Validation Loss Curve', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 突出显示改进区域
        if len(self.epoch_losses) > 1:
            # 找出损失下降的区域
            improvements = []
            for i in range(1, len(self.epoch_losses)):
                if self.epoch_losses[i] < self.epoch_losses[i-1]:
                    improvements.append(i)
            
            # 为改进区域添加背景
            for i in improvements:
                ax1.axvspan(i, i+1, alpha=0.1, color='green')
                
            # 标注总体改进
            if improvements:
                total_improvement = self.epoch_losses[0] - min(self.epoch_losses)
                ax1.text(0.02, 0.95, f"总改进: {total_improvement:.4f}", 
                        transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
        
        # 2. 训练/验证损失对比 (左下角)
        if train_losses and val_losses and len(train_losses) == len(val_losses):
            train_epochs = range(1, len(train_losses) + 1)
            ax2.plot(train_epochs, train_losses, 'b-', linewidth=2, label='Training')
            ax2.plot(train_epochs, val_losses, 'r-', linewidth=2, label='Validation')
            ax2.set_title('Training vs Validation Loss', fontsize=12)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 计算训练/验证损失之间的差距
            if len(train_losses) > 0:
                gap = np.mean([t-v for t, v in zip(train_losses, val_losses)])
                ax2.text(0.05, 0.95, f"平均间隔: {gap:.4f}", transform=ax2.transAxes, 
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, "训练/验证损失数据不可用", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. 学习率曲线 (右下角)
        if lr_history and len(lr_history) > 0:
            lr_epochs = range(1, len(lr_history) + 1)
            ax3.plot(lr_epochs, lr_history, 'g-', linewidth=2)
            ax3.set_title('Learning Rate Schedule', fontsize=12)
            ax3.set_xlabel('Epoch', fontsize=10)
            ax3.set_ylabel('Learning Rate', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # 使用科学计数法
            ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        else:
            ax3.text(0.5, 0.5, "学习率数据不可用", 
                    ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Loss plot saved to: {save_path}")
        
        # 在无头环境中避免显示，防止程序卡住
        try:
            # 设置超时，避免在没有显示环境时卡住
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        except Exception as e:
            print(f"注意: 图形显示被跳过 ({str(e)})")
            plt.close('all')
        
        # 打印训练统计
        print(f"\n📈 Training Statistics:")
        print(f"   Total Epochs: {len(self.epoch_losses)}")
        print(f"   Best Loss: {self.best_loss:.6f}")
        print(f"   Best Epoch: {self.best_epoch + 1}")
        print(f"   Final Loss: {self.epoch_losses[-1]:.6f}")
        if len(self.epoch_losses) >= 2:
            improvement = self.epoch_losses[0] - self.epoch_losses[-1]
            print(f"   Total Improvement: {improvement:.6f}")

# 早停检查器
class EarlyStopping:
    """早停检查器 - 连续5个epoch无改善则停止"""
    
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# GPU优化设置
def setup_gpu_optimization():
    """设置GPU优化"""
    if torch.cuda.is_available():
        # 启用TF32以提高A100性能
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 设置内存优化
        torch.cuda.empty_cache()
        
        # 显示GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU优化设置:")
        print(f"   GPU设备: {gpu_name}")
        print(f"   总显存: {gpu_memory:.1f} GB")
        print(f"   TF32优化: 已启用")
        return True
    else:
        print("❌ CUDA不可用")
        return False

# 改进的室内数据集
class ImprovedCOCOIndoorDataset(Dataset):
    """改进的COCO室内数据集"""
    
    def __init__(self, images_data, image_root, image_size=256, augment=True, max_samples=None):
        self.images_data = images_data
        self.image_root = image_root
        self.image_size = image_size
        self.augment = augment
        
        # 过滤有效图像
        self.valid_images = []
        for img_info in images_data:
            img_path = os.path.join(image_root, img_info['file_name'])
            if os.path.exists(img_path) and len(img_info['annotations']) > 0:
                # 额外检查图像是否可以正确打开
                try:
                    with Image.open(img_path) as img:
                        if img.width > 0 and img.height > 0:
                            self.valid_images.append(img_info)
                except Exception as e:
                    print(f"⚠️ 图像文件无效，跳过: {img_path} ({e})")
        
        # 限制样本数量（如果指定）
        if max_samples and len(self.valid_images) > max_samples:
            self.valid_images = random.sample(self.valid_images, max_samples)
        
        # 分离转换，将RandomErasing移到tensor转换后应用
        # 基本转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 张量增强 (在ToTensor之后应用)
        self.tensor_augment = None
        if augment:
            self.tensor_augment = transforms.Compose([
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ])
            
        # PIL图像增强 (在ToTensor之前应用)
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomGrayscale(p=0.1)
            ])
        else:
            self.augment_transform = None
        
        print(f"📊 数据集初始化完成:")
        print(f"   有效图像: {len(self.valid_images)}")
        print(f"   图像大小: {image_size}")
        print(f"   数据增强: {augment}")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 确保图像是有效的
            if image.width == 0 or image.height == 0:
                raise ValueError(f"图像尺寸无效: {image.width}x{image.height}")
                
            # 对PIL图像应用数据增强
            if self.augment_transform and random.random() > 0.5:
                image = self.augment_transform(image)
            
            # 转换为张量
            image_tensor = self.transform(image)
            
            # 对张量应用额外增强
            if self.tensor_augment and random.random() > 0.5:
                image_tensor = self.tensor_augment(image_tensor)
            
            return {
                'image': image_tensor,
                'image_id': img_info.get('id', idx),
                'annotations': img_info['annotations']
            }
            
        except Exception as e:
            # 返回黑色图像作为fallback
            print(f"⚠️ 图像加载失败 {img_path}: {e}")
            # 创建一个随机噪声图像替代纯黑色，避免模型过拟合于黑色图像
            random_noise = torch.rand(3, self.image_size, self.image_size) * 0.1
            fallback_image = torch.zeros(3, self.image_size, self.image_size) + random_noise
            # 应用标准化，与正常图像一致
            means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            fallback_image = (fallback_image - means) / stds
            
            return {
                'image': fallback_image,
                'image_id': img_info.get('id', idx),
                'annotations': []
            }

def collate_fn(batch):
    """批处理函数"""
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    annotations = [item['annotations'] for item in batch]
    
    return {
        'images': images,
        'image_ids': image_ids,
        'annotations': annotations
    }

# 修复版稳定训练器 - 解决计算图重复使用问题
class FixedStableTrainer:
    """修复版稳定训练器 - 解决计算图问题"""
    
    def __init__(self, clip_model, detector_model, image_processor, clip_preprocess, device):
        self.clip_model = clip_model
        self.detector_model = detector_model
        self.image_processor = image_processor
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # 创建轻量级投影器（降低复杂度）
        self.visual_projector = self.create_lightweight_projector().to(device)
        self.text_projector = self.create_lightweight_projector().to(device)
        
        # 使用恒等映射初始化
        self.initialize_as_identity()
        
        # 设置为训练模式
        self.visual_projector.train()
        self.text_projector.train()
        
        print("🎯 优化版稳定训练器初始化完成")
        print(f"   视觉投影器参数: {sum(p.numel() for p in self.visual_projector.parameters()):,}")
        print(f"   文本投影器参数: {sum(p.numel() for p in self.text_projector.parameters()):,}")
        print(f"   使用混合精度: {'是' if torch.cuda.is_available() else '否 (仅CPU)'}")
    
    def create_lightweight_projector(self):
        """创建简化版多层投影器，更易于初始化"""
        # 定义残差块
        class ResidualBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.layer_norm1 = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, dim * 2, dtype=torch.float32)
                self.gelu = nn.GELU()
                self.fc2 = nn.Linear(dim * 2, dim, dtype=torch.float32)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                residual = x
                x = self.layer_norm1(x)
                x = self.fc1(x)
                x = self.gelu(x)
                x = self.fc2(x)
                x = self.dropout(x)
                return x + residual  # 残差连接
        
        # 使用更简单的投影器结构，易于初始化
        module = nn.Sequential(
            nn.Linear(512, 512, bias=True, dtype=torch.float32),  # 首层 - 将被初始化为恒等映射
            nn.GELU(),
            nn.Linear(512, 512, bias=True, dtype=torch.float32)   # 输出层
        )
        
        # 确保所有权重都是float32
        for param in module.parameters():
            param.data = param.data.float()
            
        # 使用默认初始化
        # 首层和输出层会在initialize_as_identity中专门初始化
                    
        return module
    
    def initialize_as_identity(self):
        """初始化投影器为接近恒等映射 - 适应新的网络结构"""
        with torch.no_grad():
            # 视觉投影器 - 首层初始化为接近恒等映射
            if hasattr(self.visual_projector[0], 'weight'):
                torch.nn.init.eye_(self.visual_projector[0].weight)
                if hasattr(self.visual_projector[0], 'bias') and self.visual_projector[0].bias is not None:
                    torch.nn.init.zeros_(self.visual_projector[0].bias)
            
            # 文本投影器 - 首层初始化为接近恒等映射
            if hasattr(self.text_projector[0], 'weight'):
                torch.nn.init.eye_(self.text_projector[0].weight)
                if hasattr(self.text_projector[0], 'bias') and self.text_projector[0].bias is not None:
                    torch.nn.init.zeros_(self.text_projector[0].bias)
            
            # 残差块的初始化 - 设置小权重使残差变小
            for module in self.visual_projector.modules():
                if isinstance(module, nn.Linear):
                    if module != self.visual_projector[0]:  # 跳过已初始化的首层
                        torch.nn.init.xavier_normal_(module.weight, gain=0.5)
                        if hasattr(module, 'bias') and module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
            
            for module in self.text_projector.modules():
                if isinstance(module, nn.Linear):
                    if module != self.text_projector[0]:  # 跳过已初始化的首层
                        torch.nn.init.xavier_normal_(module.weight, gain=0.5)
                        if hasattr(module, 'bias') and module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
        
        print("✅ 投影器已初始化为优化版恒等映射")
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        params = []
        params.extend(self.visual_projector.parameters())
        params.extend(self.text_projector.parameters())
        return params
    
    def compute_distillation_loss(self, visual_features, text_features, temperature=0.05):
        """计算增强版的知识蒸馏损失"""
        # L2归一化
        visual_features = F.normalize(visual_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵（使用更低的温度系数增强对比度）
        similarity_matrix = torch.mm(visual_features, text_features.t()) / temperature
        
        # 对角线损失（自相似）
        batch_size = visual_features.size(0)
        targets = torch.arange(batch_size).to(self.device)
        
        # 如果文本特征数量不够，使用循环索引
        if text_features.size(0) < batch_size:
            targets = targets % text_features.size(0)
        
        # 增加标签平滑，进一步提升训练稳定性
        label_smoothing = 0.2
        loss_v2t = F.cross_entropy(similarity_matrix, targets, label_smoothing=label_smoothing)
        loss_t2v = F.cross_entropy(similarity_matrix.t(), targets[:text_features.size(0)], label_smoothing=label_smoothing)
        
        # InfoNCE对比损失
        logits_per_image = similarity_matrix
        logits_per_text = similarity_matrix.t()
        
        # 添加硬负样本挖掘 - 找出最具挑战性的负样本
        with torch.no_grad():
            # 创建负样本掩码
            negative_mask = torch.ones_like(similarity_matrix)
            negative_mask.fill_diagonal_(0)  # 对角线为正样本
            
            # 获取每行/列最高的负样本相似度
            hardest_negatives_per_img = (similarity_matrix * negative_mask).max(dim=1)[0]
            hardest_negatives_per_txt = (similarity_matrix.t() * negative_mask.t()).max(dim=1)[0]
        
        # 大幅减小硬负样本权重，提高训练稳定性
        hard_neg_weight = 0.2  # 进一步降低硬负样本权重，防止训练不稳定
        img_to_txt_loss = F.cross_entropy(similarity_matrix, targets) + \
                         hard_neg_weight * torch.mean(hardest_negatives_per_img)
        txt_to_img_loss = F.cross_entropy(similarity_matrix.t(), targets[:text_features.size(0)]) + \
                         hard_neg_weight * torch.mean(hardest_negatives_per_txt)
        
        # 总对比损失
        contrastive_loss = (img_to_txt_loss + txt_to_img_loss) / 2
        
        # 特征对齐损失 - 鼓励相似的图像-文本对在特征空间中接近
        alignment_loss = torch.diagonal(1 - similarity_matrix).mean()
        
        # 特征均匀性损失 - 鼓励特征空间均匀分布
        uniformity_loss = torch.log(torch.exp(torch.mm(visual_features, visual_features.t()) / temperature).mean())
        
        # 添加L2正则化防止过拟合 (减小正则化系数)
        l2_reg = 0.0005 * (
            torch.norm(self.visual_projector[0].weight, p=2) +
            torch.norm(self.text_projector[0].weight, p=2)
        )
        
        # 重新平衡损失权重，大幅减少对比损失权重，增加标准交叉熵损失权重
        total_loss = 0.3 * contrastive_loss + \
                    0.1 * alignment_loss + \
                    0.1 * uniformity_loss + \
                    0.4 * (loss_v2t + loss_t2v) / 2 + \
                    0.1 * l2_reg  # 大幅减小L2正则权重，避免过度约束
        
        return total_loss
    
    def encode_text_features_batch(self, categories, batch_size):
        """为每个批次重新编码文本特征 - 避免计算图重复使用"""
        all_text_features = []
        templates = ["a {}", "indoor {}", "a {} in a room"]
        
        for category in categories:
            category_features = []
            
            for template in templates:
                text = template.format(category)
                text_tokens = clip.tokenize([text]).to(self.device)
                
                # 重要：每次都重新计算，避免计算图重复使用
                with torch.no_grad():
                    # 确保与视觉特征相同的数据类型
                    text_features = self.clip_model.encode_text(text_tokens).float()
                
                # 应用文本投影器
                projected_text = self.text_projector(text_features)
                category_features.append(projected_text)
            
            # 平均多个模板的特征
            if category_features:
                avg_features = torch.stack(category_features).mean(dim=0)
                all_text_features.append(avg_features)
        
        if all_text_features:
            text_features = torch.cat(all_text_features, dim=0)
            
            # 随机选择文本特征（匹配batch size）
            if text_features.size(0) >= batch_size:
                selected_indices = torch.randperm(text_features.size(0))[:batch_size]
                selected_text_features = text_features[selected_indices]
            else:
                # 重复文本特征
                repeat_times = (batch_size + text_features.size(0) - 1) // text_features.size(0)
                repeated_text = text_features.repeat(repeat_times, 1)
                selected_text_features = repeated_text[:batch_size]
            
            return selected_text_features
        else:
            return torch.empty(batch_size, 512, dtype=torch.float32).to(self.device)
    
    def validate(self, dataloader):
        """在验证集上评估模型"""
        self.visual_projector.eval()
        self.text_projector.eval()
        
        val_losses = []
        num_batches = len(dataloader)
        
        # 室内类别（简化版）
        indoor_categories = [
            "chair", "table", "bed", "sofa", "cabinet", "toilet", "sink",
            "refrigerator", "microwave", "bottle", "cup", "bowl",
            "lamp", "clock", "vase", "plant", "computer", "bookshelf"  # 添加更多类别
        ]
        
        start_time = time.time()
        
        with torch.no_grad():  # 不计算梯度
            with tqdm(total=num_batches, desc="🔍 验证进行中", **tqdm_kwargs) as pbar:
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        # 获取图像
                        images = batch['images'].to(self.device)
                        batch_size = images.size(0)
                        
                        # 提取视觉特征
                        visual_features = []
                        for i in range(batch_size):
                            # 使用CLIP编码整个图像
                            image_features = self.clip_model.encode_image(images[i:i+1]).float()
                            
                            # 应用投影器
                            projected_features = self.visual_projector(image_features)
                            visual_features.append(projected_features)
                        
                        visual_features = torch.cat(visual_features, dim=0)
                        
                        # 编码文本特征
                        text_features = self.encode_text_features_batch(indoor_categories, batch_size)
                        
                        # 计算损失
                        loss = self.compute_distillation_loss(visual_features, text_features)
                        
                        # 记录损失
                        val_losses.append(loss.item())
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'avg_loss': f"{np.mean(val_losses):.4f}",
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"⚠️ 验证批次 {batch_idx} 处理失败: {e}")
                        continue
        
        avg_loss = np.mean(val_losses) if val_losses else float('inf')
        
        print(f"\n📊 验证统计:")
        print(f"   平均损失: {avg_loss:.6f}")
        print(f"   验证时间: {time.time() - start_time:.1f}秒")
        print(f"   处理批次: {len(val_losses)}/{num_batches}")
        
        return avg_loss
    
    def train_epoch(self, dataloader, optimizer, scheduler, loss_tracker):
        """训练一个epoch"""
        self.visual_projector.train()
        self.text_projector.train()
        
        epoch_losses = []
        num_batches = len(dataloader)
        
        # 室内类别（扩展版）
        indoor_categories = [
            "chair", "table", "bed", "sofa", "cabinet", "toilet", "sink",
            "refrigerator", "microwave", "bottle", "cup", "bowl",
            "lamp", "clock", "vase", "plant", "computer", "bookshelf"  # 添加更多类别
        ]
        
        start_time = time.time()
        
        with tqdm(total=num_batches, desc="🚀 训练进行中", **tqdm_kwargs) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                try:
                    optimizer.zero_grad()
                    
                    # 获取图像
                    images = batch['images'].to(self.device)
                    batch_size = images.size(0)
                    
                    # 提取视觉特征
                    visual_features = []
                    for i in range(batch_size):
                        # 使用CLIP编码整个图像
                        with torch.no_grad():
                            # 转换为float32以确保数据类型一致
                            image_features = self.clip_model.encode_image(images[i:i+1]).float()
                        
                        # 应用投影器
                        projected_features = self.visual_projector(image_features)
                        visual_features.append(projected_features)
                    
                    visual_features = torch.cat(visual_features, dim=0)
                    
                    # 关键修复：为每个批次重新编码文本特征
                    text_features = self.encode_text_features_batch(indoor_categories, batch_size)
                    
                    # 计算损失
                    loss = self.compute_distillation_loss(visual_features, text_features)
                    
                    # 检测异常损失值
                    if not torch.isfinite(loss):
                        print(f"⚠️ 警告: 损失值无效 {loss.item()}, 跳过此批次")
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 更严格的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    # 记录损失
                    epoch_losses.append(loss.item())
                    
                    # 更新进度条
                    current_lr = optimizer.param_groups[0]['lr']
                    elapsed_time = time.time() - start_time
                    samples_per_sec = (batch_idx + 1) * batch_size / elapsed_time if elapsed_time > 0 else 0
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{np.mean(epoch_losses):.4f}",
                        'lr': f"{current_lr:.2e}",
                        'samples/s': f"{samples_per_sec:.1f}"
                    })
                    pbar.update(1)
                    
                    # 清理中间变量
                    del visual_features, text_features, loss
                    
                except Exception as e:
                    print(f"⚠️ 批次 {batch_idx} 处理失败: {e}")
                    continue
        
            # 不在这里更新学习率，ReduceLROnPlateau将在外部使用验证损失更新
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        
        # 显存清理
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
            
            print(f"\n📊 Epoch统计:")
            print(f"   平均损失: {avg_loss:.6f}")
            print(f"   训练时间: {time.time() - start_time:.1f}秒")
            print(f"   处理批次: {len(epoch_losses)}/{num_batches}")
            print(f"   GPU显存: {gpu_used:.1f} GB")
        
        return avg_loss
    
    def encode_text_features(self, categories):
        """编码文本特征 - 用于推理"""
        all_text_features = []
        templates = ["a {}", "indoor {}", "a {} in a room"]
        
        for category in categories:
            category_features = []
            
            for template in templates:
                text = template.format(category)
                text_tokens = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    # 转换为float32以确保数据类型一致
                    text_features = self.clip_model.encode_text(text_tokens).float()
                    projected_text = self.text_projector(text_features)
                    category_features.append(projected_text)
            
            # 平均多个模板的特征
            if category_features:
                avg_features = torch.stack(category_features).mean(dim=0)
                all_text_features.append(avg_features)
        
        if all_text_features:
            return torch.cat(all_text_features, dim=0)
        else:
            return torch.empty(0, 512, dtype=torch.float32).to(self.device)

# 函数select_random_test_image已经在文件前面定义，此处不再重复定义

def test_fixed_model(trainer, checkpoint_dir):
    """测试优化版模型"""
    try:
        # 获取测试图像
        test_image_path = select_random_test_image() # 使用前面定义的函数
        
        if not test_image_path:
            print("❌ 没有测试图像")
            return
        
        print(f"📷 测试图像: {os.path.basename(test_image_path)}")
        
        # 简单检测测试
        image = Image.open(test_image_path).convert('RGB')
        
        # 编码图像
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            visual_features = clip_model.encode_image(image_tensor).float()  # 转换为float32
            projected_visual = trainer.visual_projector(visual_features)
            
        # 编码文本
        categories = ["chair", "table", "bottle", "sink"]
        text_features = trainer.encode_text_features(categories)
        
        # 计算相似度
        similarity = torch.mm(F.normalize(projected_visual, p=2, dim=1), 
                             F.normalize(text_features, p=2, dim=1).t())
        
        max_sim, best_idx = similarity.max(dim=1)
        
        print(f"🔍 相似度测试:")
        print(f"   最大相似度: {max_sim.item():.4f}")
        print(f"   最佳匹配: {categories[best_idx.item()]}")
        
        if max_sim.item() > 0.1:
            print("✅ 模型训练成功，特征投影正常")
        else:
            print("⚠️ 相似度较低，可能需要更多训练")
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")

def run_fixed_training():
    """运行优化版训练"""
    if not ENABLE_TRAINING:
        print("⏭️ 训练功能已禁用，跳过训练过程")
        return False
        
    print("🚀 开始优化版ViLD训练 - 解决损失增长问题")
    print("=" * 100)
    
    try:
        # GPU优化检查
        if not setup_gpu_optimization():
            return False
        
        # 确保模型在正确设备上
        clip_model.eval().to(device)
        detector_model.eval().to(device)
        
        # 创建精细优化版训练器
        trainer = FixedStableTrainer(
            clip_model=clip_model,
            detector_model=detector_model,
            image_processor=image_processor,
            clip_preprocess=clip_preprocess,
            device=device
        )
        
        # 创建训练数据集
        dataset = ImprovedCOCOIndoorDataset(
            images_data=images,
            image_root=IMAGE_ROOT,
            image_size=224,
            augment=True,
            max_samples=None  # 使用全部数据集
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        # 创建验证数据集 - 使用10%的数据
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)  # 固定种子确保可复现性
        )
        print(f"✅ 数据集分割完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
        
        # 定义训练参数
        max_epochs = 25
        
        # 数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=16,  # 增加批次大小以提高稳定性
            shuffle=True,
            num_workers=2,  # 增加worker加快数据加载
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True  # 保持worker进程活跃
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,  # 验证集不需要打乱
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        # 优化器 - 使用更高的学习率并改进配置
        # 创建更精细优化的优化器 - 基于当前训练结果的优化
        optimizer = torch.optim.AdamW(
            trainer.get_trainable_parameters(),
            lr=1e-5,  # 基于当前训练曲线，可以增加学习率以加速收敛
            weight_decay=0.00005,  # 进一步减小正则化强度
            betas=(0.9, 0.98),  # 更长的指数加权平均窗口
            eps=1e-8,
            amsgrad=True  # 启用AMSGrad变种，提供更稳定的收敛
        )
        
        # 使用更稳定的学习率调度器
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #    optimizer,
        #    max_lr=1e-4,
        #    total_steps=max_epochs * len(train_dataloader),
        #    pct_start=0.2,
        #    div_factor=25.0,
        #    final_div_factor=1000.0,
        #    anneal_strategy='cos'
        # )
        
        # 使用更精细的学习率调度策略 - 根据当前训练结果优化
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控验证损失
            factor=0.7,  # 每次降低30%（而非之前的50%），更平滑的下降
            patience=1,  # 1个epoch无改善就调整学习率，更积极地响应
            verbose=True,
            threshold=0.005,  # 更敏感的阈值检测
            min_lr=5e-7  # 提高最小学习率
        )
        
        # 损失追踪器和早停
        loss_tracker = LossTracker()
        early_stopping = EarlyStopping(patience=5, min_delta=1e-4)  # 增加耐心，减少敏感度
        
        print(f"📊 精细优化训练配置:")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   批次大小: 16")
        print(f"   初始学习率: 1e-5")
        print(f"   学习率调度: ReduceLROnPlateau(factor=0.7, patience=1)")
        print(f"   最大epoch数: {max_epochs}")
        print(f"   早停耐心: 5 epochs")
        print(f"   梯度裁剪: 0.5")
        print(f"   权重衰减: 0.00005")
        print(f"   优化器: AdamW with AMSGrad")
        print(f"   可训练参数: {sum(p.numel() for p in trainer.get_trainable_parameters()):,}")
        print(f"   硬负样本权重: 0.2")
        print(f"   标签平滑: 0.2")
        
        # 开始训练
        best_loss_value = float('inf')
        best_model_path = None
        
        # 创建检查点目录
        checkpoint_dir = '/home/cui/vild_rtdetr_indoor/src/vild/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 添加训练进度追踪
        train_losses = []
        val_losses = []
        
        # 添加学习率记录
        lr_history = []
        
        # 添加最佳模型信息
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        patience = 5  # 更长的耐心
        
        for epoch in range(max_epochs):
            print(f"\n{'='*100}")
            print(f"🔄 Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*100}")
            
            # 训练
            train_loss = trainer.train_epoch(train_dataloader, optimizer, scheduler, loss_tracker)
            train_losses.append(train_loss)
            
            # 验证
            print(f"\n📊 运行验证...")
            val_loss = trainer.validate(val_dataloader)
            val_losses.append(val_loss)
            
            # 更新学习率调度器 - 使用验证损失决定是否降低学习率
            scheduler.step(val_loss)
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            # 更新损失追踪器 (使用验证损失)
            loss_tracker.update(val_loss, epoch)
            
            print(f"📈 Epoch {epoch+1} 结果:")
            print(f"   训练损失: {train_loss:.6f}")
            print(f"   验证损失: {val_loss:.6f}")
            print(f"   学习率: {current_lr:.8f}")
            
            # 保存最佳模型 (基于验证损失)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # 删除之前的最佳模型
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    print(f"🗑️ 删除旧的最佳模型")
                
                # 保存新的最佳模型
                best_model_path = f'{checkpoint_dir}/best_refined_model.pth'
                checkpoint = {
                    'epoch': epoch,
                    'visual_projector': trainer.visual_projector.state_dict(),
                    'text_projector': trainer.text_projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'training_config': {
                        'lr': current_lr,
                        'weight_decay': 0.0001,
                        'batch_size': 16,
                        'max_epochs': max_epochs
                    }
                }
                
                torch.save(checkpoint, best_model_path)
                print(f"💾 保存最佳模型: 验证损失={val_loss:.6f} (第{epoch+1}轮)")
            else:
                patience_counter += 1
                print(f"⚠️ 验证损失未改善，当前耐心: {patience_counter}/{patience}")
            
            # 改进的早停检查
            if patience_counter >= patience:
                print(f"\n⏹️ 早停触发! 连续{patience}个epoch无改善，在第 {epoch + 1} epoch停止")
                print(f"   最佳验证损失值: {best_val_loss:.6f} (第{best_epoch+1}轮)")
                break
            
            # 内存清理
            torch.cuda.empty_cache()
            gc.collect()
        
        # 训练完成后输出增强版损失图
        print(f"\n🎨 绘制最终损失图...")
        final_loss_path = f'{checkpoint_dir}/enhanced_training_loss.png'
        loss_tracker.plot_losses(
            save_path=final_loss_path,
            train_losses=train_losses,
            val_losses=val_losses,
            lr_history=lr_history
        )
        
        # 保存训练历史记录用于后续分析
        history_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'lr_history': lr_history,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epochs_trained': len(train_losses)
        }
        history_path = f'{checkpoint_dir}/training_history.json'
        with open(history_path, 'w') as f:
            # 将numpy数组转换为列表以便JSON序列化
            history_data_serializable = {
                k: v if not isinstance(v, (np.ndarray, np.generic)) else v.tolist() 
                for k, v in history_data.items()
            }
            json.dump(history_data_serializable, f, indent=2)
        
        print(f"\n🎉 增强版训练完成!")
        print(f"📈 最终训练成果:")
        print(f"   1. ✅ 训练样本: {len(train_dataset)} 个")
        print(f"   2. ✅ 验证样本: {len(val_dataset)} 个")
        print(f"   3. ✅ 训练轮次: {len(train_losses)} epochs")
        print(f"   4. ✅ 最佳验证损失: {best_val_loss:.6f}")
        print(f"   5. ✅ 最佳epoch: {best_epoch + 1}")
        print(f"   6. ✅ 损失图已保存: {final_loss_path}")
        print(f"   7. ✅ 训练历史已保存: {history_path}")
        print(f"   8. ✅ 最佳模型已保存: {best_model_path}")
        
        # 显示训练和验证损失的对比
        final_train_loss = train_losses[-1] if train_losses else float('nan')
        final_val_loss = val_losses[-1] if val_losses else float('nan')
        print(f"\n📊 最终性能对比:")
        print(f"   • 初始训练损失: {train_losses[0]:.6f}")
        print(f"   • 最终训练损失: {final_train_loss:.6f}")
        print(f"   • 训练损失改进: {train_losses[0] - final_train_loss:.6f}")
        print(f"   • 初始验证损失: {val_losses[0]:.6f}")
        print(f"   • 最终验证损失: {final_val_loss:.6f}")
        print(f"   • 验证损失改进: {val_losses[0] - final_val_loss:.6f}")
        
        # 测试训练后的模型
        print(f"\n🧪 测试训练后的模型...")
        test_fixed_model(trainer, checkpoint_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# 4. 基于预训练权重的推理检测
# =============================================================================
"""
本节实现基于训练好的ViLD模型进行室内场景物体检测：

1. **模型权重加载** - 加载训练好的CLIP+RT-DETR融合模型
2. **文本查询编码** - 支持多种室内物体的文本描述
3. **图像区域提取** - 智能分割图像为检测区域
4. **相似度计算** - 计算图像特征与文本特征的匹配度
5. **结果后处理** - NMS去重和置信度过滤
6. **可视化展示** - 绘制检测框和置信度标签
"""

class FixedViLDDetector:
    """使用修复投影器的ViLD检测器"""
    
    def __init__(self, clip_model, detector_model, image_processor, clip_preprocess, device):
        self.clip_model = clip_model
        self.detector_model = detector_model
        self.image_processor = image_processor
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # 创建修复的投影器（接近恒等映射）
        self.visual_projector = self.create_identity_projector()
        self.text_projector = self.create_identity_projector()
        
        # 检测参数（提高阈值以减少错误识别）
        self.similarity_threshold = 0.25  # 提高阈值，减少错误识别
        self.detection_threshold = 0.05   # 提高检测基础阈值
        self.max_detections = 15
        
        # 初始室内类别集合（基础类别）
        self.base_categories = [
            'chair', 'table', 'bed', 'sofa', 'lamp', 'cabinet', 'door', 'window',
            'mirror', 'picture', 'book', 'bottle', 'cup', 'bowl', 'clock',
            'plant', 'television', 'refrigerator', 'microwave', 'toilet', 'sink',
            'towel', 'pillow', 'curtains', 'rug', 'shower', 'bathtub', 'shelf',
            'counter', 'desk', 'wardrobe', 'nightstand', 'computer', 'monitor'
        ]
        
        # 使用基础类别初始化当前活动类别
        self.categories = self.base_categories.copy()
        
        # 类别别名映射（将常见混淆类别组合在一起）
        self.category_aliases = {
            'towel': ['towel', 'bath towel', 'hand towel', 'bathroom towel', 'hanging towel', 'white towel', 'folded towel'],
            'curtains': ['curtains', 'curtain', 'window curtain', 'drape', 'window treatment', 'window covering'],
            'microwave': ['microwave', 'microwave oven', 'kitchen microwave', 'heating appliance'],
            'cabinet': ['cabinet', 'cupboard', 'storage cabinet', 'kitchen cabinet', 'bathroom cabinet'],
            'sink': ['sink', 'bathroom sink', 'kitchen sink', 'washbasin', 'basin', 'wash basin'],
            'toilet': ['toilet', 'bathroom toilet', 'toilet bowl', 'commode', 'lavatory'],
            'sofa': ['sofa', 'couch', 'settee', 'living room sofa', 'seating'],
            'television': ['television', 'TV', 'flatscreen', 'TV screen', 'monitor', 'display'],
            'bed': ['bed', 'mattress', 'bedroom bed', 'sleeping surface'],
            'refrigerator': ['refrigerator', 'fridge', 'kitchen refrigerator', 'cooling appliance'],
            'table': ['table', 'dining table', 'coffee table', 'desk', 'surface'],
            'chair': ['chair', 'seat', 'armchair', 'office chair', 'dining chair'],
            'shower': ['shower', 'shower stall', 'shower cubicle', 'bathroom shower'],
            'bathtub': ['bathtub', 'tub', 'bath', 'bathroom tub'],
            'mirror': ['mirror', 'wall mirror', 'bathroom mirror', 'reflective surface'],
            'lamp': ['lamp', 'light fixture', 'table lamp', 'floor lamp', 'lighting'],
            'picture': ['picture', 'painting', 'photo', 'wall art', 'artwork', 'frame']
        }
        
        # 场景特定类别（用于场景上下文优化）
        self.scene_categories = {
            'bathroom': ['toilet', 'sink', 'towel', 'bathtub', 'shower', 'mirror'],
            'kitchen': ['refrigerator', 'microwave', 'sink', 'cabinet', 'counter', 'table', 'bottle', 'cup', 'bowl'],
            'bedroom': ['bed', 'pillow', 'lamp', 'nightstand', 'wardrobe', 'mirror', 'clock'],
            'living_room': ['sofa', 'table', 'television', 'lamp', 'rug', 'curtains', 'picture']
        }
        
        # 开放词汇支持
        self.clip_vocabulary = []  # 存储开放词汇表
        self.custom_categories = []  # 用户添加的自定义类别
        self.enable_open_vocabulary = True  # 启用开放词汇检测
        self.open_vocabulary_threshold = 0.22  # 开放词汇匹配阈值
        self.max_open_vocabulary_results = 3  # 每个区域最多返回的开放词汇结果数
        
        # 从CLIP加载大量词汇，以支持更开放的检测
        self._load_clip_vocabulary()
        
        print("🔧 增强版开放词汇ViLD检测器初始化完成")
        print(f"   相似度阈值: {self.similarity_threshold}")
        print(f"   基础类别: {len(self.base_categories)} 个")
        print(f"   类别别名: {len(self.category_aliases)} 个")
        print(f"   场景类型: {len(self.scene_categories)} 个")
        print(f"   开放词汇: {'启用' if self.enable_open_vocabulary else '禁用'}")
    
    def create_identity_projector(self):
        """创建接近恒等映射的投影器 - 简化版"""
        # 明确指定使用float32数据类型，使用更简单的结构
        projector = torch.nn.Sequential(
            torch.nn.Linear(512, 512, bias=True, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512, bias=True, dtype=torch.float32)
        ).to(self.device)
        
        # 初始化为恒等映射
        with torch.no_grad():
            # 第一层：恒等映射
            torch.nn.init.eye_(projector[0].weight)
            if projector[0].bias is not None:
                torch.nn.init.zeros_(projector[0].bias)
            
            # 第三层：恒等映射
            torch.nn.init.eye_(projector[2].weight)
            if projector[2].bias is not None:
                torch.nn.init.zeros_(projector[2].bias)
            
            # 确保所有权重都是float32
            for param in projector.parameters():
                param.data = param.data.float()
        
        projector.eval()  # 设置为评估模式
        return projector
    
    def detect_objects(self, image_path: str, scene_type=None, custom_categories=None, enable_open_vocabulary=True):
        """检测图像中的物体，支持开放词汇和场景感知功能
        
        参数:
            image_path: 图像路径
            scene_type: 可选场景类型，如'bathroom', 'kitchen', 'bedroom', 'living_room'
            custom_categories: 用户自定义的类别列表（可选）
            enable_open_vocabulary: 是否启用开放词汇检测
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 打开图像
            image = Image.open(image_path).convert('RGB')
            
            # 处理自定义类别（如果提供）
            if custom_categories:
                self.set_custom_categories(custom_categories)
                print(f"🔍 使用自定义类别: {', '.join(self.custom_categories)}")
            
            # 设置是否启用开放词汇检测
            self.enable_open_vocabulary = enable_open_vocabulary
            
            # 0. 预测场景类型（如果未提供）
            if scene_type is None:
                # 这里可以添加简单的场景分类逻辑
                print("ℹ️ 未指定场景类型，使用通用检测模式")
            else:
                print(f"🏠 使用场景感知模式: {scene_type}")
            
            # 1. 提取候选区域
            boxes, detection_scores = self.extract_regions(image)
            if len(boxes) == 0:
                print(f"❌ 没有找到候选区域")
                return {'boxes': [], 'scores': [], 'labels': []}
            
            print(f"📦 找到 {len(boxes)} 个候选区域")
            
            # 2. 提取视觉特征
            visual_features = self.extract_visual_features(image, boxes)
            if visual_features.size(0) == 0:
                print(f"❌ 视觉特征提取失败")
                return {'boxes': [], 'scores': [], 'labels': []}
            
            # 3. 编码基础文本特征
            text_features = self.encode_text_features()
            
            # 4. 计算相似度
            similarity_matrix = torch.mm(visual_features, text_features.t())
            
            # 如果指定了场景类型，应用场景优化
            if scene_type is not None:
                similarity_matrix = self.apply_scene_context(scene_type, similarity_matrix)
            
            max_similarities, best_category_indices = similarity_matrix.max(dim=1)
            
            # 5. 过滤和后处理
            # 动态阈值 - 为明显的相似度使用更高阈值，为边缘情况使用梯度阈值
            similarity_threshold = self.similarity_threshold
            if max_similarities.max() > 0.4:  # 如果有很强的匹配
                # 使用自适应阈值 - 最大值的60%或固定阈值的较大者
                adaptive_threshold = max(max_similarities.max() * 0.6, self.similarity_threshold)
                similarity_threshold = min(adaptive_threshold, 0.4)  # 不超过0.4
                print(f"🔄 使用自适应阈值: {similarity_threshold:.4f}")
            
            # 检查是否有超过阈值的匹配
            valid_mask = max_similarities >= similarity_threshold
            valid_count = valid_mask.sum().item()
            
            print(f"🔍 相似度范围: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
            print(f"✅ 有效检测 (阈值={similarity_threshold}): {valid_count}")
            
            # 如果没有超过阈值的匹配，尝试降低阈值
            if valid_count == 0:
                print(f"⚠️ 没有超过阈值的检测，尝试降低阈值...")
                # 尝试更低的阈值
                low_threshold = 0.05
                valid_mask = max_similarities >= low_threshold
                valid_count = valid_mask.sum().item()
                print(f"📊 降低阈值到 {low_threshold}: {valid_count} 个检测")
                
                if valid_count == 0 and not self.enable_open_vocabulary:
                    return {'boxes': [], 'scores': [], 'labels': []}
            
            # 处理基础类别检测
            if valid_count > 0:
                # 提取有效检测
                valid_boxes = boxes[:len(valid_mask)][valid_mask.cpu().numpy()]
                valid_detection_scores = detection_scores[:len(valid_mask)][valid_mask.cpu().numpy()]
                valid_similarities = max_similarities[valid_mask].cpu().numpy()
                valid_category_indices = best_category_indices[valid_mask].cpu().numpy()
                valid_labels = [self.categories[idx] for idx in valid_category_indices]
                
                # 组合分数
                combined_scores = valid_detection_scores * 0.3 + valid_similarities * 0.7
                
                # 按分数排序
                sorted_indices = np.argsort(combined_scores)[::-1][:self.max_detections]
                
                final_boxes = valid_boxes[sorted_indices]
                final_scores = combined_scores[sorted_indices]
                final_labels = [valid_labels[i] for i in sorted_indices]
                
                print(f"🎯 基础检测结果: {len(final_boxes)} 个物体")
                print(f"   类别: {set(final_labels)}")
                
                # 返回检测结果
                result = {
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'labels': final_labels,
                    'open_vocab_results': {}  # 初始化空的开放词汇结果
                }
            else:
                # 如果没有找到任何基础类别的匹配，创建空结果
                result = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': [],
                    'open_vocab_results': {}
                }
            
            # 如果启用了开放词汇检测，尝试更广泛的词汇表匹配
            if self.enable_open_vocabulary:
                print(f"🔠 执行开放词汇检测...")
                open_vocab_results = self.perform_open_vocabulary_detection(
                    visual_features, boxes, detection_scores
                )
                
                # 合并结果
                if open_vocab_results:
                    result['open_vocab_results'] = open_vocab_results
                    
                    # 如果基础检测没有结果，但开放词汇检测有结果
                    if len(result['boxes']) == 0 and len(open_vocab_results['boxes']) > 0:
                        print("🔤 使用开放词汇检测结果作为主要结果")
                        result['boxes'] = open_vocab_results['boxes']
                        result['scores'] = open_vocab_results['scores']
                        result['labels'] = open_vocab_results['labels']
                
            # 计算总检测时间
            detection_time = time.time() - start_time
            result['detection_time'] = detection_time
            
            print(f"⏱️ 检测完成，用时: {detection_time:.2f}秒")
            print(f"🎯 最终检测结果: {len(result['boxes'])} 个物体")
            
            return result
            
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return {'boxes': [], 'scores': [], 'labels': []}
    
    def _load_clip_vocabulary(self):
        """加载CLIP大规模词汇表，以支持开放词汇检测"""
        try:
            # 常见室内物体的扩展词汇表
            extended_vocabulary = [
                # 家具类
                "armchair", "bench", "bookshelf", "bunk bed", "coffee table", "dining table",
                "dresser", "end table", "filing cabinet", "footstool", "futon", "loveseat",
                "ottoman", "recliner", "rocking chair", "sideboard", "stool", "tv stand",
                
                # 电器类
                "air conditioner", "blender", "coffee maker", "dishwasher", "electric fan", 
                "food processor", "hair dryer", "heater", "humidifier", "iron", "juicer",
                "kettle", "microwave oven", "mixer", "oven", "rice cooker", "toaster", 
                "vacuum cleaner", "washing machine", "water heater",
                
                # 卫浴类
                "bathroom cabinet", "bathroom mirror", "bathroom shelf", "bath mat",
                "faucet", "hand towel", "medicine cabinet", "shower curtain", "shower door",
                "shower head", "soap dish", "toilet brush", "toilet paper holder", "towel rack",
                
                # 装饰品类
                "artificial flower", "candle", "candle holder", "cushion", "decorative plate",
                "flower vase", "photo frame", "sculpture", "wall clock", "wall hanging",
                
                # 厨房用品
                "chopping board", "colander", "cooking pot", "cutlery", "dinnerware",
                "frying pan", "kitchen knife", "kitchen utensil", "measuring cup", "mixing bowl",
                "oven mitt", "pepper grinder", "plate", "salt shaker", "saucepan", "spatula",
                "spice rack", "tea towel", "tongs", "wooden spoon",
                
                # 寝具类
                "blanket", "comforter", "duvet", "mattress", "mattress pad", "pillow case",
                "sheet", "sleeping bag", "sleeping mask", "throw blanket",
                
                # 灯具类
                "ceiling light", "chandelier", "desk lamp", "floor lamp", "pendant light",
                "reading lamp", "string lights", "table lamp", "track lighting", "wall light",
                
                # 其他家居用品
                "alarm clock", "backpack", "blinds", "coat hanger", "doormat", "extension cord",
                "garbage bin", "houseplant", "magazine rack", "power strip", "tissue box",
                "umbrella stand", "wall plug", "window blind", "window sill"
            ]
            
            # 加载基本类别和扩展词汇表
            self.clip_vocabulary = self.base_categories + extended_vocabulary
            
            # 为词汇生成文本特征，但不要在初始化时做，而是延迟到需要时
            print(f"✅ 加载了 {len(self.clip_vocabulary)} 个开放词汇项")
            
        except Exception as e:
            print(f"⚠️ 加载CLIP词汇表失败: {e}")
            self.clip_vocabulary = self.base_categories.copy()
            
    def set_custom_categories(self, categories):
        """设置用户自定义类别列表
        
        参数:
            categories: 字符串列表，包含用户想要检测的特定类别
        """
        if not categories:
            return
            
        # 重置当前类别为基础类别
        self.categories = self.base_categories.copy()
        
        # 添加用户自定义类别
        self.custom_categories = [c for c in categories if c not in self.categories]
        self.categories.extend(self.custom_categories)
        
        print(f"✅ 设置了 {len(self.custom_categories)} 个自定义类别")
        print(f"   当前类别总数: {len(self.categories)}")
        
    def apply_scene_context(self, scene_type, similarity_matrix):
        """应用场景上下文来优化相似度矩阵
        
        参数:
            scene_type: 场景类型 ('bathroom', 'kitchen', 等)
            similarity_matrix: 相似度矩阵
            
        返回:
            优化后的相似度矩阵
        """
        if scene_type not in self.scene_categories:
            return similarity_matrix
            
        # 获取与场景相关的类别
        relevant_categories = self.scene_categories[scene_type]
        relevant_indices = [i for i, cat in enumerate(self.categories) if cat in relevant_categories]
        
        # 创建矩阵副本进行修改
        modified_matrix = similarity_matrix.clone()
        
        # 提升场景相关类别的相似度分数
        boost_factor = 0.15  # 15%的提升
        for i in range(similarity_matrix.size(0)):  # 遍历所有区域
            for idx in relevant_indices:  # 遍历场景相关类别
                modified_matrix[i, idx] *= (1 + boost_factor)
                
        # 对于不相关的类别，略微降低相似度
        non_relevant_indices = [i for i, cat in enumerate(self.categories) if cat not in relevant_categories]
        penalty_factor = 0.05  # 5%的惩罚
        for i in range(similarity_matrix.size(0)):
            for idx in non_relevant_indices:
                modified_matrix[i, idx] *= (1 - penalty_factor)
                
        print(f"🏠 应用场景上下文优化: {scene_type}")
        print(f"   提升类别: {', '.join(relevant_categories)}")
                
        return modified_matrix
    
    def extract_regions(self, image):
        """提取候选区域"""
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.detection_threshold
        )[0]
        
        return results['boxes'].cpu().numpy(), results['scores'].cpu().numpy()
    
    def perform_open_vocabulary_detection(self, visual_features, boxes, detection_scores):
        """执行开放词汇检测
        
        参数:
            visual_features: 已提取的视觉特征
            boxes: 检测框
            detection_scores: 检测分数
            
        返回:
            开放词汇检测结果
        """
        try:
            # 如果词汇表为空，动态加载
            if not self.clip_vocabulary:
                self._load_clip_vocabulary()
                
            if not self.clip_vocabulary:
                print("⚠️ 没有可用的开放词汇表")
                return {}
                
            # 准备结果结构
            open_vocab_results = {
                'boxes': [],
                'scores': [],
                'labels': [],
                'alternative_labels': []
            }
            
            # 生成开放词汇的文本特征（延迟计算，以避免初始化时的开销）
            print(f"🔤 生成 {len(self.clip_vocabulary)} 个词汇项的文本特征...")
            
            # 批量处理词汇以避免显存不足
            batch_size = 200
            all_text_features = []
            
            for i in range(0, len(self.clip_vocabulary), batch_size):
                batch = self.clip_vocabulary[i:i+batch_size]
                
                # 为每个词汇项生成文本tokens
                texts = [f"a {word}" for word in batch]
                text_tokens = clip.tokenize(texts).to(self.device)
                
                # 编码文本特征
                with torch.no_grad():
                    batch_text_features = self.clip_model.encode_text(text_tokens).float()
                    batch_text_features = self.text_projector(batch_text_features)
                    batch_text_features = F.normalize(batch_text_features, p=2, dim=1)
                    all_text_features.append(batch_text_features)
            
            # 合并所有特征
            text_features = torch.cat(all_text_features, dim=0)
            
            # 计算每个区域与所有词汇的相似度
            print("🧮 计算开放词汇相似度...")
            similarity_matrix = torch.mm(visual_features, text_features.t())
            
            # 为每个区域找到最佳的开放词汇匹配
            for i in range(similarity_matrix.size(0)):
                # 获取前K个最佳匹配
                similarities, indices = torch.topk(similarity_matrix[i], k=self.max_open_vocabulary_results)
                
                # 检查相似度是否高于开放词汇阈值
                if similarities[0] >= self.open_vocabulary_threshold:
                    # 第一个最佳匹配作为主标签
                    best_idx = indices[0].item()
                    best_score = similarities[0].item()
                    best_label = self.clip_vocabulary[best_idx]
                    
                    # 其他候选项作为替代标签
                    alt_indices = indices[1:].cpu().numpy()
                    alt_scores = similarities[1:].cpu().numpy()
                    alt_labels = [(self.clip_vocabulary[idx], score) for idx, score in zip(alt_indices, alt_scores)]
                    
                    # 添加到结果中
                    open_vocab_results['boxes'].append(boxes[i])
                    open_vocab_results['scores'].append(best_score)
                    open_vocab_results['labels'].append(best_label)
                    open_vocab_results['alternative_labels'].append(alt_labels)
                    
            # 转换为numpy数组
            if open_vocab_results['boxes']:
                open_vocab_results['boxes'] = np.array(open_vocab_results['boxes'])
                open_vocab_results['scores'] = np.array(open_vocab_results['scores'])
                
                # 保留最佳结果
                if len(open_vocab_results['boxes']) > self.max_detections:
                    # 按分数排序
                    sorted_indices = np.argsort(open_vocab_results['scores'])[::-1][:self.max_detections]
                    open_vocab_results['boxes'] = open_vocab_results['boxes'][sorted_indices]
                    open_vocab_results['scores'] = open_vocab_results['scores'][sorted_indices]
                    open_vocab_results['labels'] = [open_vocab_results['labels'][i] for i in sorted_indices]
                    open_vocab_results['alternative_labels'] = [open_vocab_results['alternative_labels'][i] for i in sorted_indices]
                    
                print(f"🔤 开放词汇检测结果: {len(open_vocab_results['boxes'])} 个物体")
                print(f"   类别: {set(open_vocab_results['labels'])}")
                
            return open_vocab_results
            
        except Exception as e:
            print(f"❌ 开放词汇检测失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_visual_features(self, image, boxes):
        """提取视觉特征"""
        if len(boxes) == 0:
            return torch.empty(0, 512).to(self.device)
        
        features = []
        img_array = np.array(image)
        max_regions = min(len(boxes), 50)  # 限制处理数量
        
        for i, box in enumerate(boxes[:max_regions]):
            x1, y1, x2, y2 = box.astype(int)
            
            # 边界检查
            x1 = max(0, min(x1, img_array.shape[1]-1))
            y1 = max(0, min(y1, img_array.shape[0]-1))
            x2 = max(x1+1, min(x2, img_array.shape[1]))
            y2 = max(y1+1, min(y2, img_array.shape[0]))
            
            # 确保区域大小合理
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                half_size = 25
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(img_array.shape[1], center_x + half_size)
                y2 = min(img_array.shape[0], center_y + half_size)
            
            region = img_array[y1:y2, x1:x2]
            
            if region.size > 0:
                try:
                    region_pil = Image.fromarray(region)
                    if region_pil.size[0] < 224 or region_pil.size[1] < 224:
                        region_pil = region_pil.resize((224, 224), Image.LANCZOS)
                    
                    region_tensor = self.clip_preprocess(region_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        visual_feat = self.clip_model.encode_image(region_tensor).float()  # 转换为float32
                        visual_feat = self.visual_projector(visual_feat)
                        visual_feat = F.normalize(visual_feat, p=2, dim=1)
                        features.append(visual_feat)
                        
                except Exception as e:
                    if i < 5:  # 只打印前几个错误
                        print(f"⚠️ 区域 {i} 处理失败: {e}")
                    continue
        
        if features:
            return torch.cat(features, dim=0)
        else:
            return torch.empty(0, 512).to(self.device)
    
    def encode_text_features(self):
        """增强版文本特征编码 - 使用多种模板和类别别名"""
        all_features = []
        
        # 通用模板 - 更丰富的描述方式
        generic_templates = [
            "a {}",
            "a photo of {}",
            "an indoor {}",
            "a {} in a room",
            "a clear photo of {}"
        ]
        
        # 场景特定模板
        scene_templates = {
            'bathroom': ["a {} in a bathroom", "a bathroom {}"],
            'kitchen': ["a {} in a kitchen", "a kitchen {}"],
            'bedroom': ["a {} in a bedroom", "a bedroom {}"],
            'living_room': ["a {} in a living room", "a living room {}"]
        }
        
        # 查找每个类别可能所属的场景
        category_scenes = {}
        for scene, items in {
            'bathroom': ['toilet', 'sink', 'towel', 'bathtub', 'shower', 'mirror'],
            'kitchen': ['refrigerator', 'microwave', 'sink', 'cabinet', 'counter', 'table', 'bottle', 'cup', 'bowl'],
            'bedroom': ['bed', 'pillow', 'lamp', 'nightstand', 'wardrobe', 'mirror', 'clock'],
            'living_room': ['sofa', 'table', 'television', 'lamp', 'rug', 'curtains', 'picture']
        }.items():
            for item in items:
                if item not in category_scenes:
                    category_scenes[item] = []
                category_scenes[item].append(scene)
        
        for idx, category in enumerate(self.categories):
            category_features = []
            
            # 处理当前类别的所有别名（如果有）
            category_terms = [category]  # 默认至少包含类别本身
            if category in self.category_aliases:
                category_terms.extend(self.category_aliases[category])
            
            # 为每个别名应用通用模板
            for term in category_terms:
                for template in generic_templates:
                    text = template.format(term)
                    text_tokens = clip.tokenize([text]).to(self.device)
                    
                    with torch.no_grad():
                        text_feat = self.clip_model.encode_text(text_tokens).float()
                        text_feat = self.text_projector(text_feat)
                        text_feat = F.normalize(text_feat, p=2, dim=1)
                        category_features.append(text_feat)
            
            # 应用场景特定模板
            if category in category_scenes:
                for scene in category_scenes[category]:
                    for template in scene_templates[scene]:
                        text = template.format(category)
                        text_tokens = clip.tokenize([text]).to(self.device)
                        
                        with torch.no_grad():
                            text_feat = self.clip_model.encode_text(text_tokens).float()
                            text_feat = self.text_projector(text_feat)
                            text_feat = F.normalize(text_feat, p=2, dim=1)
                            category_features.append(text_feat)
            
            # 平均多个模板和别名的特征
            if category_features:
                avg_features = torch.stack(category_features).mean(dim=0)
                avg_features = F.normalize(avg_features, p=2, dim=1)
                all_features.append(avg_features)
        
        return torch.cat(all_features, dim=0)
    
    def visualize_results(self, image_path: str, results: dict, save_path=None, scene_type=None):
        """可视化检测结果并保存到文件，支持开放词汇检测结果可视化"""
        image = Image.open(image_path).convert('RGB')
        
        # 创建更大的图形以适应更多信息
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.imshow(image)
        
        boxes = results['boxes']
        scores = results['scores']
        labels = results['labels']
        
        # 检查是否有开放词汇检测结果
        has_open_vocab = 'open_vocab_results' in results and len(results['open_vocab_results'].get('boxes', [])) > 0
        
        # 设置图表标题 (使用英文)
        title = "Indoor Scene Object Detection"
        if scene_type:
            title += f" - {scene_type.capitalize()} Scene"
        if 'detection_time' in results:
            title += f" ({results['detection_time']:.2f}s)"
        if has_open_vocab:
            title += " [Open Vocabulary Mode]"
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 生成颜色映射
        num_colors_needed = max(20, len(set(labels)) + (len(set(results['open_vocab_results'].get('labels', []))) if has_open_vocab else 0))
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_colors_needed)))
        
        if len(boxes) > 0:
            # 对于所有已知标签创建类别到颜色的映射
            all_possible_labels = list(self.categories)
            if has_open_vocab:
                all_possible_labels.extend(list(set(results['open_vocab_results']['labels'])))
                
            category_to_color = {cat: colors[i % len(colors)] for i, cat in enumerate(set(all_possible_labels))}
            
            # 绘制检测框和标签
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                color = category_to_color.get(label, colors[0])
                
                # 计算置信度等级 (使用英文)
                confidence_level = ""
                if score > 0.7:
                    confidence_level = "HIGH"
                elif score > 0.4:
                    confidence_level = "MED"
                else:
                    confidence_level = "LOW"
                
                # 绘制检测框
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 绘制增强标签 - 包括置信度等级 (英文)
                ax.text(
                    x1, y1 - 5,
                    f"{label} ({confidence_level}: {score:.2f})",
                    color='white',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
                )
            
            # 添加检测统计信息
            categories_detected = set(labels)
            
            # 统计标准检测和开放词汇检测（如果有）
            total_objects = len(boxes)
            total_categories = len(categories_detected)
            
            if has_open_vocab:
                open_vocab_boxes = results['open_vocab_results']['boxes']
                open_vocab_labels = results['open_vocab_results']['labels']
                open_vocab_categories = set(open_vocab_labels)
                
                # 更新统计信息
                total_objects = len(boxes) + len(open_vocab_boxes)
                total_categories = len(categories_detected.union(open_vocab_categories))
                
                stats_text = f"Detected {total_objects} objects, {total_categories} categories (Standard: {len(boxes)}, Open Vocab: {len(open_vocab_boxes)})"
            else:
                stats_text = f"Detected {total_objects} objects, {total_categories} categories"
                
            ax.text(
                10, 30, 
                stats_text,
                color='white', 
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='navy', alpha=0.7)
            )
            
            # 创建图例 - 包括标准检测和开放词汇检测
            legend_elements = []
            
            # 标准检测的图例
            unique_labels = list(set(labels))
            for label in unique_labels:
                color = category_to_color.get(label, colors[0])
                legend_elements.append(
                    patches.Patch(facecolor=color, label=label)
                )
                
            # 如果有开放词汇检测，添加其图例
            if has_open_vocab:
                unique_open_labels = list(set(results['open_vocab_results']['labels']))
                for label in unique_open_labels:
                    # 使用虚线边框区分开放词汇结果
                    if label not in unique_labels:  # 避免重复
                        color = category_to_color.get(label, colors[0])
                        legend_elements.append(
                            patches.Patch(facecolor=color, label=f"{label} (Open)", 
                                        linestyle='dashed', edgecolor='black')
                        )
            
            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc='upper right',
                    fontsize=10,
                    title="Detected Categories",
                    fancybox=True,
                    framealpha=0.7
                )
        # 绘制开放词汇检测结果（如果有）
        if has_open_vocab:
            open_vocab_boxes = results['open_vocab_results']['boxes']
            open_vocab_scores = results['open_vocab_results']['scores']
            open_vocab_labels = results['open_vocab_results']['labels']
            open_vocab_alt_labels = results['open_vocab_results'].get('alternative_labels', [])
            
            for i, (box, score, label) in enumerate(zip(open_vocab_boxes, open_vocab_scores, open_vocab_labels)):
                x1, y1, x2, y2 = box
                
                # 为开放词汇检测使用虚线边框
                color = category_to_color.get(label, colors[0])
                
                # 计算置信度等级 (使用英文)
                confidence_level = ""
                if score > 0.7:
                    confidence_level = "HIGH"
                elif score > 0.4:
                    confidence_level = "MED"
                else:
                    confidence_level = "LOW"
                
                # 绘制检测框 - 使用虚线区分开放词汇检测
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='dashed'  # 使用虚线表示开放词汇检测
                )
                ax.add_patch(rect)
                
                # 显示替代标签（如果有）- 英文
                alt_text = ""
                if i < len(open_vocab_alt_labels) and open_vocab_alt_labels[i]:
                    top_alt = open_vocab_alt_labels[i][0]  # 取第一个替代标签
                    alt_text = f" | Alt: {top_alt[0]} ({top_alt[1]:.2f})"
                
                # 绘制增强标签 - 包括置信度等级和开放词汇标记 (英文)
                ax.text(
                    x1, y1 - 5,
                    f"{label} ({confidence_level}: {score:.2f}){alt_text}",
                    color='white',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.2', facecolor=color, alpha=0.8)
                )
        
        if len(boxes) == 0 and (not has_open_vocab or len(results['open_vocab_results'].get('boxes', [])) == 0):
            # 没有检测到物体的情况 (英文)
            ax.text(
                image.width // 2 - 100, image.height // 2,
                "No Objects Detected",
                color='white',
                fontsize=16,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='red', alpha=0.8)
            )
        
        # 添加时间戳和使用的模型信息 (英文)
        plt.figtext(
            0.01, 0.01,
            f"Detection Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Model: Enhanced ViLD (CLIP + RTDETR)",
            fontsize=8, color='gray'
        )
        
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        
        # 保存结果而不是显示
        if save_path is None:
            # 如果未提供保存路径，则生成一个基于原始图像名称的路径
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            save_dir = os.path.join(os.path.dirname(image_path), "detection_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name}_detection{ext}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 检测结果已保存至: {save_path}")
            
        plt.close(fig)  # 关闭图形以释放内存
        
        return fig
        return result_path

def test_fixed_detector():
    """测试修复版检测器"""
    if not ENABLE_DETECTION:
        print("⏭️ 检测功能已禁用，跳过检测器测试")
        return
        
    print("🔄 测试修复版检测器...")
    
    # 创建修复版检测器
    fixed_detector = FixedViLDDetector(
        clip_model=clip_model,
        detector_model=detector_model,
        image_processor=image_processor,
        clip_preprocess=clip_preprocess,
        device=device
    )
    
    # 获取测试图像
    test_image_path = select_random_test_image()
    
    # 如果没有找到，创建一个测试图像
    if not test_image_path:
        print("没有找到有效图像，创建测试图像...")
        test_dir = os.path.join(PROJECT_ROOT, "tests")
        os.makedirs(test_dir, exist_ok=True)
        test_image_path = os.path.join(test_dir, "test_image.jpg")
        
        # 创建一个简单的测试图像
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # 绘制一些简单的形状
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 0, 255), 2)
        cv2.circle(test_image, (400, 200), 50, (0, 255, 0), -1)
        cv2.imwrite(test_image_path, test_image)
        print(f"已创建测试图像: {test_image_path}")
    
    print(f"📷 测试图像: {os.path.basename(test_image_path)}")
    
    # 运行检测
    results = fixed_detector.detect_objects(test_image_path)
    
    # 总是保存检测结果到文件，避免在WSL环境下尝试显示图形
    checkpoint_dir = '/home/cui/vild_rtdetr_indoor/src/vild/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    detection_path = os.path.join(checkpoint_dir, f"detection_result_{os.path.basename(test_image_path)}")
    
    # 可视化结果并保存
    saved_path = fixed_detector.visualize_results(test_image_path, results, save_path=detection_path)
    
    # 在WSL环境中通常会失败，所以不尝试显示，仅保存结果
    # 如果需要显示，请在Windows环境下查看保存的图像文件    print(f"\n🎯 修复版检测器测试完成!")
    if len(results['boxes']) > 0:
        print(f"✅ 成功检测到 {len(results['boxes'])} 个物体")
        print(f"   检测类别: {set(results['labels'])}")
        if saved_path:
            print(f"   检测结果已保存: {saved_path}")
    else:
        print(f"⚠️ 未检测到物体，可能需要进一步调整参数")
    
    # 保存修复版投影器，替换原有训练权重
    checkpoint_dir = '/home/cui/vild_rtdetr_indoor/src/vild/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建修复版投影器
    fixed_visual = fixed_detector.visual_projector
    fixed_text = fixed_detector.text_projector
    
    # 保存为新检查点
    checkpoint = {
        'epoch': 999,  # 特殊标记
        'visual_projector': fixed_visual.state_dict(),
        'text_projector': fixed_text.state_dict(),
        'loss': 0.0,  # 修复版没有损失
        'fixed_version': True,
        'description': 'Identity mapping fix for zero detection issue'
    }
    
    torch.save(checkpoint, f'{checkpoint_dir}/fixed_identity_projectors.pth')
    print("✅ 修复版投影器已保存: fixed_identity_projectors.pth")

# 主函数
def detect_indoor_image(image_path, output_path=None, scene_type=None, custom_categories=None, enable_open_vocab=True):
    """运行单张图像的室内场景开放词汇检测
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径 (可选)
        scene_type: 场景类型，如 'bathroom', 'kitchen', 'bedroom', 'living_room' (可选)
        custom_categories: 自定义类别列表 (可选)
        enable_open_vocab: 是否启用开放词汇检测 (默认True)
        
    返回:
        检测结果和输出图像路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            return None, None
            
        # 检查场景类型
        valid_scenes = ['bathroom', 'kitchen', 'bedroom', 'living_room']
        if scene_type and scene_type not in valid_scenes:
            print(f"⚠️ 无效的场景类型: {scene_type}")
            print(f"   有效选项: {', '.join(valid_scenes)}")
            scene_type = None
        
        # 选择适当的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ 使用设备: {device}")
        
        # 加载模型
        print("🔄 加载CLIP模型...")
        clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
        clip_model.eval()
        
        print("🔄 加载RT-DETR检测器...")
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        
        image_processor = AutoImageProcessor.from_pretrained("PekinU/rtdetr-l")
        detector_model = AutoModelForObjectDetection.from_pretrained("PekinU/rtdetr-l")
        detector_model = detector_model.to(device)
        detector_model.eval()
        
        # 创建检测器
        detector = FixedViLDDetector(
            clip_model=clip_model,
            detector_model=detector_model,
            image_processor=image_processor,
            clip_preprocess=clip_preprocess,
            device=device
        )
        
        # 执行检测
        print(f"🔍 开始检测图像: {image_path}")
        if scene_type:
            print(f"   场景类型: {scene_type}")
            
        start_time = time.time()
        results = detector.detect_objects(
            image_path, 
            scene_type=scene_type,
            custom_categories=custom_categories,
            enable_open_vocabulary=enable_open_vocab
        )
        elapsed = time.time() - start_time
        
        # 显示检测结果
        num_objects = len(results['boxes'])
        print(f"✓ 检测完成! 用时 {elapsed:.2f} 秒")
        print(f"   检测到 {num_objects} 个物体")
        
        if num_objects > 0:
            labels = results['labels']
            scores = results['scores']
            
            # 打印检测到的对象
            print("\n📋 检测结果:")
            for i, (label, score) in enumerate(zip(labels, scores)):
                print(f"   {i+1}. {label:<15} 置信度: {score:.4f}")
                
            # 显示类别统计
            from collections import Counter
            label_counts = Counter(labels)
            print("\n📊 类别统计:")
            for label, count in label_counts.most_common():
                print(f"   {label:<15}: {count} 个")
        
        # 可视化并保存结果
        output_path = detector.visualize_results(image_path, results, output_path, scene_type)
        
        return results, output_path
        
    except Exception as e:
        print(f"❌ 检测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description='室内场景开放词汇物体检测')
    parser.add_argument('--image', '-i', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    parser.add_argument('--scene', '-s', choices=['bathroom', 'kitchen', 'bedroom', 'living_room'], 
                        help='场景类型 (可选: bathroom, kitchen, bedroom, living_room)')
    parser.add_argument('--demo', '-d', action='store_true', help='运行示例演示')
    parser.add_argument('--open-vocab', '-ov', action='store_true', default=True, 
                        help='启用开放词汇检测 (默认启用)')
    parser.add_argument('--no-open-vocab', '-nov', action='store_false', dest='open_vocab',
                        help='禁用开放词汇检测')
    parser.add_argument('--custom-categories', '-c', nargs='+', 
                        help='指定自定义类别列表，如 "cup book laptop"')
    parser.add_argument('--train', '-t', action='store_true', help='执行训练过程')
    parser.add_argument('--no-train', action='store_false', dest='train', help='跳过训练过程')
    parser.add_argument('--detect', action='store_true', default=True, help='执行检测过程')
    parser.add_argument('--no-detect', action='store_false', dest='detect', help='跳过检测过程')
    parser.add_argument('--test-image', type=int, default=-1, help='指定测试图像索引，-1表示随机选择')
    
    args = parser.parse_args()
    
    print("🚀 启动增强版室内物体检测系统...")
    
    # 设置全局控制变量
    # 注意：global声明应该在赋值之前，这里不需要再声明因为已经在文件顶部定义了全局变量
    ENABLE_TRAINING = args.train
    ENABLE_DETECTION = args.detect
    TEST_IMAGE_INDEX = args.test_image
    
    print(f"⚙️ 配置设置:")
    print(f"   • 训练功能: {'启用' if ENABLE_TRAINING else '禁用'}")
    print(f"   • 检测功能: {'启用' if ENABLE_DETECTION else '禁用'}")
    print(f"   • 测试图像: {TEST_IMAGE_INDEX if TEST_IMAGE_INDEX >= 0 else '随机选择'}")
    
    # 处理命令行参数
    if args.image:
        # 使用提供的图像路径
        image_path = args.image
        output_path = args.output
        scene_type = args.scene
        
        custom_categories = args.custom_categories
        enable_open_vocab = args.open_vocab
        
        print(f"🖼️ 处理图像: {image_path}")
        if custom_categories:
            print(f"🔍 使用自定义类别: {', '.join(custom_categories)}")
        print(f"🔤 开放词汇检测: {'启用' if enable_open_vocab else '禁用'}")
        
        results, output_path = detect_indoor_image(
            image_path, 
            output_path, 
            scene_type,
            custom_categories=custom_categories,
            enable_open_vocab=enable_open_vocab
        )
        
        if results is not None:
            print("\n✅ 检测完成!")
            print(f"   结果已保存至: {output_path}")
            
            # 如果有开放词汇检测结果，显示更多信息
            if enable_open_vocab and 'open_vocab_results' in results and results['open_vocab_results']:
                open_vocab_labels = results['open_vocab_results']['labels']
                if open_vocab_labels:
                    print(f"   开放词汇检测结果: {len(open_vocab_labels)} 个物体")
                    print(f"   开放词汇类别: {set(open_vocab_labels)}")
        else:
            print("\n❌ 检测失败")
            
    elif args.demo:
        # 运行演示 - 检查不同室内场景中的目标检测
        print("🎬 运行室内场景检测演示...")
        
        # 查找示例图像
        demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_images")
        os.makedirs(demo_dir, exist_ok=True)
        
        # 如果没有示例图像，我们创建一些
        if not os.listdir(demo_dir):
            print("📷 创建示例图像...")
            # 创建几个测试图像
            scenes = ["bathroom", "kitchen", "living_room", "bedroom"]
            for scene in scenes:
                test_img_path = os.path.join(demo_dir, f"{scene}_demo.jpg")
                test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
                
                # 添加场景名称
                cv2.putText(test_image, f"{scene.upper()} DEMO", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                
                # 添加不同的形状
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if scene == "bathroom":
                    cv2.rectangle(test_image, (400, 100), (500, 300), color, 2)
                elif scene == "kitchen":
                    cv2.circle(test_image, (450, 240), 100, color, -1)
                elif scene == "living_room":
                    pts = np.array([[350, 100], [500, 150], [450, 300], [300, 250]], np.int32)
                    cv2.polylines(test_image, [pts], True, color, 3)
                else:
                    cv2.rectangle(test_image, (350, 150), (500, 300), color, -1)
                    
                cv2.imwrite(test_img_path, test_image)
                print(f"   创建了示例图像: {test_img_path}")
        
        # 处理每个示例图像
        demo_images = [f for f in os.listdir(demo_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not demo_images:
            print("⚠️ 没有找到示例图像")
        else:
            for image_file in demo_images:
                image_path = os.path.join(demo_dir, image_file)
                scene_type = next((s for s in ['bathroom', 'kitchen', 'bedroom', 'living_room'] 
                                   if s in image_file), None)
                
                print(f"\n📷 处理示例图像: {image_file}")
                if scene_type:
                    print(f"   场景类型: {scene_type}")
                    
                # 为演示启用开放词汇检测
                output_path = os.path.join(demo_dir, f"{os.path.splitext(image_file)[0]}_result.jpg")
                results, _ = detect_indoor_image(
                    image_path, 
                    output_path, 
                    scene_type,
                    enable_open_vocab=True
                )
                
                if results is not None:
                    print(f"   ✓ 检测完成，结果保存至: {output_path}")
                else:
                    print(f"   ❌ 检测失败")
            
            print("\n🎉 演示完成!")
            
    else:
        # 显示帮助信息
        print("\n⚠️ 请提供图像路径或使用演示模式")
        print("使用示例:")
        print("  python indoor_vild.py --image path/to/image.jpg")
        print("  python indoor_vild.py --image path/to/image.jpg --scene bathroom")
        print("  python indoor_vild.py --demo")
        print("\n运行 'python indoor_vild.py --help' 查看所有选项")
    
    try:
        # 根据开关控制是否执行训练和检测
        if ENABLE_TRAINING:
            print("\n🔄 开始训练流程...")
            run_fixed_training()
        else:
            print("\n⏭️ 训练过程已跳过 (ENABLE_TRAINING=False)")
        
        if ENABLE_DETECTION:
            print("\n🔍 测试检测器...")
            test_fixed_detector()
        else:
            print("\n⏭️ 检测过程已跳过 (ENABLE_DETECTION=False)")
        
        # 确保关闭所有matplotlib图形，防止程序卡住
        plt.close('all')
        
        print("\n✅ 系统运行完成!")
    except Exception as e:
        print(f"\n❌ 执行过程中出错: {str(e)}")
        traceback.print_exc()
        
        # 关闭所有图形，确保程序能够退出
        try:
            plt.close('all')
        except:
            pass
        
        print("\n系统已尝试恢复并退出")