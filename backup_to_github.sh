#!/bin/bash
"""
RT-DETR项目完整备份脚本
整理项目文件并准备提交到GitHub
"""

set -e  # 遇到错误立即停止

PROJECT_DIR="/home/cui/rtdetr_indoor"
BACKUP_DIR="/home/cui/rtdetr_backup_$(date +%Y%m%d_%H%M%S)"
AUTODL_DIR="/root/autodl-tmp"

echo "🗄️ RT-DETR项目完整备份工具"
echo "=" * 50

# 创建备份目录
mkdir -p "$BACKUP_DIR"
cd "$PROJECT_DIR"

echo "📁 项目基本信息:"
echo "   项目目录: $PROJECT_DIR"
echo "   备份目录: $BACKUP_DIR"
echo "   项目大小: $(du -sh . | cut -f1)"

# 1. 复制主项目代码
echo "📋 1. 复制主项目代码..."
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.cache' \
    --exclude='runs/detect' \
    --exclude='*.tmp' \
    --exclude='datasets/coco/images/' \
    --exclude='datasets/coco/labels/' \
    --exclude='datasets/coco_indoor/images/' \
    --exclude='datasets/coco_indoor/labels/' \
    --exclude='datasets/homeobjects_extended_yolo_smart/images/' \
    --exclude='datasets/homeobjects_extended_yolo_smart/labels/' \
    --exclude='*.jpg' \
    --exclude='*.png' \
    --exclude='*.jpeg' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    "$PROJECT_DIR/" "$BACKUP_DIR/rtdetr_indoor/"

# 2. 跳过权重文件(太大,不适合GitHub)
echo "📋 2. 跳过权重文件备份..."
echo "   ⚠️ 权重文件已排除(*.pt, *.pth文件太大)"
echo "   💡 如需备份权重,请使用Git LFS或云存储"

# 3. 数据集配置文件已包含在主项目中
echo "📋 3. 数据集配置..."
echo "   ✅ YAML配置文件已包含"
echo "   ⚠️ 图片和标注文件已排除(数据集太大)"

# 4. 创建完整的项目说明
cat > "$BACKUP_DIR/BACKUP_README.md" << 'EOF'
# RT-DETR HomeObjects 项目备份

## 📁 目录结构
```
rtdetr_backup_YYYYMMDD_HHMMSS/
├── rtdetr_indoor/           # 主项目代码
├── weights/                 # 预训练权重
├── trained_weights/         # 训练产生的权重
├── dataset_info/           # 数据集描述信息
└── BACKUP_README.md        # 本文件
```

## 🚀 项目特性
- RT-DETR模型在HomeObjects数据集上的训练
- 三种模型变体对比 (RT-DETR-L, RT-DETR-MNV4, RT-DETR-MNV4-SEA)
- RTX 4090 GPU优化配置
- 完整的训练脚本和分析工具

## 📊 主要文件
- `scripts/training/auto_train_rtdetr.py` - 自动训练脚本
- `filter_homeobjects_smart.py` - 智能数据集筛选
- `MODEL_COMPARISON_ANALYSIS.md` - 模型对比分析
- `FINAL_REPORT.md` - 项目总结报告

## 🔄 恢复使用
1. 恢复主项目: `cp -r rtdetr_indoor/ /path/to/workspace/`
2. 恢复权重: `cp weights/* /path/to/workspace/`
3. 安装依赖: `pip install -r rtdetr_indoor/requirements.txt`
4. 配置数据集路径
5. 运行训练: `./start_auto_training.sh`

## ⚙️ 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 12.1+
- RTX 4090 (推荐)
- 32GB+ 系统内存

EOF

# 5. 生成文件清单
echo "📋 5. 生成文件清单..."
find "$BACKUP_DIR" -type f > "$BACKUP_DIR/file_list.txt"
echo "文件总数: $(wc -l < "$BACKUP_DIR/file_list.txt")"

# 7. 计算备份大小
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "📊 备份完成!"
echo "   备份位置: $BACKUP_DIR"
echo "   备份大小: $BACKUP_SIZE"

# 8. GitHub准备
echo "📋 6. 准备GitHub仓库..."
cd "$BACKUP_DIR"

# 初始化git仓库
git init
git config user.name "cui123000"
git config user.email "your-email@example.com"  # 请替换为你的邮箱

# 创建.gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# 大文件
*.pt
*.pth
*.onnx
*.bin

# 数据
datasets/images/
datasets/labels/
*.jpg
*.png
*.jpeg

# 临时文件
.cache/
.tmp/
*.tmp
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

EOF

# 添加文件
git add .
git commit -m "Initial commit: RT-DETR HomeObjects项目完整备份

项目特性:
- RT-DETR模型训练和优化
- HomeObjects数据集智能筛选 
- 三模型变体性能对比
- RTX 4090 GPU优化配置
- NaN损失问题修复
- 完整训练脚本和分析工具

主要组件:
- 自动训练脚本
- 模型分析工具  
- 数据集处理工具
- 性能对比报告
- 环境配置文件"

echo "✅ Git仓库初始化完成"
echo ""
echo "🚀 接下来的步骤:"
echo "1. 在GitHub创建新仓库 (建议名称: rtdetr-homeobjects)"
echo "2. 添加远程仓库:"
echo "   cd $BACKUP_DIR"
echo "   git remote add origin https://github.com/cui123000/rtdetr-homeobjects.git"
echo "3. 推送到GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "⚠️ 注意: 大文件(权重文件)已被.gitignore排除"
echo "   如需分享权重，请使用Git LFS或其他方式"

echo ""
echo "🎯 备份完成! 总大小: $BACKUP_SIZE"