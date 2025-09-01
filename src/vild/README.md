# 增强版室内物体检测系统

基于Vision-Language知识蒸馏(ViLD)的开放世界室内物体检测系统，集成了CLIP和RT-DETR模型。

## 功能特点

- 开放词汇目标检测：不限于预定义类别
- 场景感知：针对不同室内场景优化检测结果
- 多模态特征融合：结合视觉和语言特征
- 可视化增强：直观展示检测结果和置信度

## 使用方法

### 基本参数

```bash
# 基本用法
python indoor_vild\ copy.py [选项]

# 指定输入图像
python indoor_vild\ copy.py --image path/to/image.jpg

# 指定场景类型
python indoor_vild\ copy.py --image path/to/image.jpg --scene bathroom

# 运行演示模式
python indoor_vild\ copy.py --demo
```

### 新增功能参数

```bash
# 控制训练开关
python indoor_vild\ copy.py --train       # 执行训练
python indoor_vild\ copy.py --no-train    # 跳过训练

# 控制检测开关
python indoor_vild\ copy.py --detect      # 执行检测
python indoor_vild\ copy.py --no-detect   # 跳过检测

# 指定测试图像
python indoor_vild\ copy.py --test-image 5    # 使用索引为5的图像进行测试
python indoor_vild\ copy.py --test-image -1   # 随机选择测试图像
```

### 完整选项

```
--image, -i            输入图像路径
--output, -o           输出图像路径
--scene, -s            场景类型 (bathroom, kitchen, bedroom, living_room)
--demo, -d             运行示例演示
--open-vocab, -ov      启用开放词汇检测 (默认)
--no-open-vocab, -nov  禁用开放词汇检测
--custom-categories, -c 指定自定义类别列表
--train, -t            执行训练过程
--no-train             跳过训练过程
--detect               执行检测过程 (默认)
--no-detect            跳过检测过程
--test-image           指定测试图像索引，-1表示随机选择
```

## 示例场景

- 浴室场景 (`--scene bathroom`)：识别浴缸、淋浴、毛巾架等
- 厨房场景 (`--scene kitchen`)：识别冰箱、微波炉、水槽等
- 卧室场景 (`--scene bedroom`)：识别床、衣柜、床头柜等
- 客厅场景 (`--scene living_room`)：识别沙发、电视、咖啡桌等
