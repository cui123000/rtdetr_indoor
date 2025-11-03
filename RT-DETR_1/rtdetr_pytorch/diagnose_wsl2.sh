#!/bin/bash
# WSL2 RT-DETR 训练诊断和预热脚本

echo "=========================================="
echo "🔍 RT-DETR WSL2 环境诊断"
echo "=========================================="

# 1. 检查GPU
echo -e "\n📊 1. GPU 信息检查"
echo "----------------------------------------"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader

# 2. 检查系统资源
echo -e "\n💻 2. 系统资源检查"
echo "----------------------------------------"
echo "CPU核心数: $(nproc)"
echo "总内存: $(free -h | awk '/^Mem:/ {print $2}')"
echo "可用内存: $(free -h | awk '/^Mem:/ {print $7}')"
echo "磁盘空间: $(df -h /home | awk 'NR==2 {print $4}')"

# 3. 检查Python环境
echo -e "\n🐍 3. Python 环境检查"
echo "----------------------------------------"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# 4. 检查配置文件
echo -e "\n⚙️  4. 训练配置检查"
echo "----------------------------------------"
CONFIG="/home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch/configs/rtdetr/include/dataloader.yml"
echo "batch_size (train): $(grep -A 2 'train_dataloader:' $CONFIG | grep 'batch_size:' | awk '{print $2}')"
echo "num_workers (train): $(grep -A 3 'train_dataloader:' $CONFIG | grep 'num_workers:' | awk '{print $2}')"

# 5. GPU预热测试
echo -e "\n🔥 5. GPU 预热测试（降低初始负载冲击）"
echo "----------------------------------------"
python << 'PYTHON_EOF'
import torch
import torch.nn as nn
import time

print("开始GPU预热...")

# 创建一个小模型预热GPU
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU()
).cuda()

# 小批量数据预热
for i in range(5):
    x = torch.randn(2, 3, 640, 640).cuda()
    with torch.cuda.amp.autocast():
        y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"  预热 {i+1}/5 完成")
    time.sleep(1)

# 清理
del model, x, y, loss
torch.cuda.empty_cache()
print("✅ GPU预热完成，显存已清理")
PYTHON_EOF

echo -e "\n=========================================="
echo "✅ 诊断完成"
echo "=========================================="
echo ""
echo "💡 如果训练还是立即崩溃，请尝试："
echo "   1. 在Windows中降低GPU性能模式"
echo "   2. 限制WSL2内存 (创建 .wslconfig)"
echo "   3. 更新NVIDIA驱动"
echo "   4. 检查Windows事件查看器中的错误日志"
echo ""
