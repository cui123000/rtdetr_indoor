# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - README
"""

# ViLD 模块化实现

这个项目是对基于ViLD（Vision-Language Knowledge Distillation）的开放世界室内物体检测系统的模块化实现。ViLD结合了CLIP（Contrastive Language-Image Pre-training）模型和RT-DETR（Real-Time Detection Transformer）检测器，实现对室内场景中物体的检测和分类。

## 项目结构

```
vild_modular/
├── main.py         - 程序入口点，包含命令行界面
├── config.py       - 配置文件，包含模型和训练参数
├── model.py        - 模型定义和加载逻辑
├── data_loader.py  - 数据加载和处理
├── detector.py     - 对象检测实现
├── training.py     - 训练逻辑
└── utils.py        - 通用工具函数
```

## 主要功能

1. **检测模式**：使用预训练模型对输入图像进行室内物体检测
2. **训练模式**：使用CLIP模型特征进行知识蒸馏训练

## 使用方法

### 安装依赖

```bash
pip install torch torchvision clip opencv-python matplotlib pillow tqdm
```

### 检测模式

检测单张图像：

```bash
python main.py --mode detect --input /path/to/image.jpg --show --save
```

检测一个目录中的所有图像：

```bash
python main.py --mode detect --input /path/to/image/directory --save --output_dir results
```

### 训练模式

使用自定义数据集进行训练：

```bash
python main.py --mode train --train_data /path/to/annotations.json --image_root /path/to/images
```

## 核心模块说明

### 1. 模型 (model.py)

- `ViLDModel`：结合CLIP和RT-DETR的主模型类
- `load_models`：加载预训练CLIP和RT-DETR模型

### 2. 数据加载器 (data_loader.py)

- `ImprovedCOCOIndoorDataset`：加载和处理COCO格式的室内场景数据集
- `load_images_data`：从JSON文件加载图像数据
- `create_dataloader`：创建训练和验证数据加载器

### 3. 检测器 (detector.py)

- `FixedViLDDetector`：使用CLIP进行开放世界物体分类的检测器
- `extract_regions`：从RT-DETR提取区域proposals
- `compute_similarity`：计算视觉和文本特征之间的相似度

### 4. 训练 (training.py)

- `StableTrainer`：稳定的知识蒸馏训练器
- `LossTracker`：训练损失追踪和可视化
- `run_fixed_training`：运行训练过程的主函数

### 5. 工具 (utils.py)

- 图像处理和可视化函数
- 非极大值抑制
- 检测结果统计和保存

## 配置 (config.py)

项目配置文件包含：
- 模型路径
- 训练参数
- 推理参数
- 室内类别分类

## 性能优化

- 使用非极大值抑制过滤重叠检测
- 实现批处理以提高推理速度
- 使用梯度裁剪和学习率调度以提高训练稳定性

## 自定义与扩展

可以通过修改`config.py`中的类别列表来扩展或定制检测的物体类别。例如：

```python
CATEGORIES = {
    'furniture': ["chair", "table", "bed", ...]
    'electronics': ["tv", "computer", "laptop", ...],
    # 添加更多类别
}
```
