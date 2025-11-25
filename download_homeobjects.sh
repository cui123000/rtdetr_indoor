#!/bin/bash
# 下载和解压 HomeObjects-3K 数据集

echo "============================================"
echo "下载 HomeObjects-3K 数据集"
echo "============================================"

DATASET_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip"
DATASET_DIR="/home/cjj/rtdetr_indoor/datasets"
DATASET_NAME="homeobjects-3K"

# 创建数据集目录
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "下载位置: $DATASET_DIR"
echo "数据集: $DATASET_NAME"
echo ""

# 检查是否已存在
if [ -d "$DATASET_DIR/$DATASET_NAME" ]; then
    echo "✓ 数据集已存在: $DATASET_DIR/$DATASET_NAME"
    ls -lah "$DATASET_DIR/$DATASET_NAME/"
    exit 0
fi

# 检查是否已下载zip文件
if [ ! -f "$DATASET_NAME.zip" ]; then
    echo "📥 开始下载... (390 MB)"
    echo "URL: $DATASET_URL"
    echo ""
    
    # 使用wget下载，支持断点续传
    wget -c "$DATASET_URL" -O "$DATASET_NAME.zip"
    
    if [ $? -ne 0 ]; then
        echo "❌ 下载失败！"
        echo "可能原因:"
        echo "  1. 网络连接问题"
        echo "  2. URL已更改"
        echo ""
        echo "替代方案: 手动下载后放在 $DATASET_DIR/ 目录"
        exit 1
    fi
else
    echo "✓ ZIP文件已存在: $DATASET_NAME.zip"
fi

# 检查文件大小
FILE_SIZE=$(stat -f%z "$DATASET_NAME.zip" 2>/dev/null || stat -c%s "$DATASET_NAME.zip" 2>/dev/null || echo "unknown")
echo "文件大小: $FILE_SIZE 字节"
echo ""

# 解压
echo "📦 解压中... (可能需要1-2分钟)"
unzip -q "$DATASET_NAME.zip"

if [ $? -eq 0 ]; then
    echo "✓ 解压成功！"
    
    # 验证
    if [ -d "$DATASET_NAME" ]; then
        echo ""
        echo "✓ 数据集已就绪"
        echo ""
        echo "数据集结构:"
        ls -lah "$DATASET_NAME/" | head -20
        echo ""
        
        # 统计文件数
        TRAIN_COUNT=$(find "$DATASET_NAME/images/train" -type f 2>/dev/null | wc -l)
        VAL_COUNT=$(find "$DATASET_NAME/images/val" -type f 2>/dev/null | wc -l)
        
        echo "📊 数据集统计:"
        echo "  训练集: $TRAIN_COUNT 张图片"
        echo "  验证集: $VAL_COUNT 张图片"
        echo "  总计: $((TRAIN_COUNT + VAL_COUNT)) 张图片"
        echo ""
        
        # 删除zip文件
        echo "清理下载文件..."
        rm -f "$DATASET_NAME.zip"
        echo "✓ 完成"
        
        echo ""
        echo "============================================"
        echo "✅ HomeObjects-3K 数据集就绪！"
        echo "============================================"
        echo ""
        echo "下一步: 使用该数据集训练"
        echo "命令: cd /home/cjj/rtdetr_indoor && bash train_homeobjects.sh"
    else
        echo "❌ 解压后未找到数据集目录"
        exit 1
    fi
else
    echo "❌ 解压失败！"
    exit 1
fi
