#!/bin/bash
"""
简化的GitHub提交脚本
直接从当前项目推送到GitHub
"""

echo "🚀 RT-DETR项目GitHub提交工具"
echo "=" * 50

# 进入项目目录
cd /home/cui/rtdetr_indoor

# 检查git状态
if [ ! -d ".git" ]; then
    echo "❌ 未发现git仓库，请先初始化"
    exit 1
fi

echo "📋 当前项目状态:"
echo "   分支: $(git branch --show-current)"
echo "   提交数: $(git rev-list --count HEAD)"

# 更新.gitignore确保不提交大文件
echo "📝 更新.gitignore..."
cat > .gitignore << 'EOF'
# Python缓存
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.cache/

# 大文件 - 权重文件
*.pt
*.pth
*.onnx
*.bin
*.safetensors

# 数据集 - 太大不适合git
datasets/homeobjects_extended_yolo_smart/images/
datasets/homeobjects_extended_yolo_smart/labels/
datasets/**/*.jpg
datasets/**/*.png
datasets/**/*.jpeg

# 临时和日志文件
*.tmp
*.log
logs/
runs/detect/
.autodl/

# IDE和编辑器
.vscode/
.idea/
*.swp
*.swo
*~

# 系统文件
.DS_Store
Thumbs.db

# 大型第三方库 (保留配置)
# RT-DETR_1/ # 如果太大可以取消注释

# 训练输出 (太大)
/root/autodl-tmp/

EOF

# 添加重要文件
echo "📁 添加项目文件..."
git add .
git status

# 创建详细的提交信息
COMMIT_MSG="feat: RT-DETR HomeObjects项目完整版本

🎯 项目特性:
- RT-DETR模型在HomeObjects数据集上的训练优化
- 三种模型变体性能对比 (RT-DETR-L, MNV4, MNV4-SEA)  
- RTX 4090 GPU专用优化配置
- NaN损失问题完整修复方案
- 智能数据集筛选和分析工具

📊 主要组件:
- scripts/training/auto_train_rtdetr.py - 自动训练脚本
- filter_homeobjects_smart.py - 数据集筛选工具
- MODEL_COMPARISON_ANALYSIS.md - 详细性能分析
- tools/measure_models.py - 模型测量工具
- start_auto_training.sh - 一键启动脚本

🔧 技术突破:
- 解决了梯度爆炸导致的NaN损失问题
- 优化学习率调度 (0.0001起始 + 10epoch预热)
- 禁用AMP避免数值不稳定
- 数据集从24K优化到7.6K高质量样本

💾 数据集: 
- HomeObjects扩展版 (智能筛选)
- 21个核心室内物体类别
- 7,634张高质量标注图像
- 95%+室内场景纯度

⚡ 性能结果:
- RT-DETR-L: 32.97M参数, 40.55ms推理
- RT-DETR-MNV4: 24.98M参数, 40.86ms推理  
- RT-DETR-MNV4-SEA: 29.06M参数, 54.72ms推理

🛠️ 环境支持:
- Python 3.8+ / PyTorch 2.0+
- CUDA 12.1+ / RTX 4090优化
- Ultralytics框架集成
- TMux会话管理"

# 提交
echo "💾 提交到本地仓库..."
git commit -m "$COMMIT_MSG"

echo ""
echo "✅ 本地提交完成!"
echo ""
echo "🔄 接下来推送到GitHub:"
echo "1. 确保GitHub仓库已创建"
echo "2. 运行推送命令:"
echo "   git push origin main"
echo ""
echo "📊 项目统计:"
echo "   文件数: $(find . -name .git -prune -o -type f -print | wc -l)"
echo "   代码行数: $(find . -name .git -prune -o -name '*.py' -exec wc -l {} + | tail -1)"
echo "   项目大小: $(du -sh --exclude=.git . | cut -f1)"

echo ""
echo "🎉 准备完成! 现在可以推送到GitHub了。"