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
    "$PROJECT_DIR/" "$BACKUP_DIR/rtdetr_indoor/"

# 2. 复制关键训练权重
echo "📋 2. 备份关键权重文件..."
mkdir -p "$BACKUP_DIR/weights"
if [ -f "$PROJECT_DIR/rtdetr-l.pt" ]; then
    cp "$PROJECT_DIR/rtdetr-l.pt" "$BACKUP_DIR/weights/"
    echo "   ✅ rtdetr-l.pt (预训练权重)"
fi

if [ -f "$PROJECT_DIR/yolo11n.pt" ]; then
    cp "$PROJECT_DIR/yolo11n.pt" "$BACKUP_DIR/weights/"
    echo "   ✅ yolo11n.pt (对比权重)"
fi

# 3. 备份训练结果
echo "📋 3. 备份训练结果..."
if [ -d "$AUTODL_DIR/rtdetr_weights" ]; then
    mkdir -p "$BACKUP_DIR/trained_weights"
    rsync -av --progress "$AUTODL_DIR/rtdetr_weights/" "$BACKUP_DIR/trained_weights/"
    echo "   ✅ 训练权重已备份"
else
    echo "   ⚠️ 未找到训练权重"
fi

# 4. 创建数据集描述文件(不包含图片)
echo "📋 4. 创建数据集描述..."
mkdir -p "$BACKUP_DIR/dataset_info"

# 数据集统计信息
if [ -d "$PROJECT_DIR/datasets" ]; then
    echo "数据集信息:" > "$BACKUP_DIR/dataset_info/dataset_summary.txt"
    find "$PROJECT_DIR/datasets" -name "*.yaml" -exec cp {} "$BACKUP_DIR/dataset_info/" \;
    find "$PROJECT_DIR/datasets" -type f | wc -l >> "$BACKUP_DIR/dataset_info/file_count.txt"
    du -sh "$PROJECT_DIR/datasets" >> "$BACKUP_DIR/dataset_info/dataset_size.txt"
    echo "   ✅ 数据集信息已保存"
fi

# 5. 创建完整的项目说明
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

# 6. 生成文件清单
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