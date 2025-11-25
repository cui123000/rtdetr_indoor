#!/bin/bash
# RT-DETR 后台训练启动脚本 - 支持所有8个模型

set -e

cd /home/cjj/rtdetr_indoor

# 显示可用模型
show_models() {
    cat << 'EOF'
📋 可用模型列表 (支持 11 个模型):

1️⃣  模型1: rtdetr-l.yaml
    - 名称: RT-DETR-L
    - batch: 24, lr: 0.0004, epochs: 100
    - 预计显存: 21GB
    - 预计时间: 2-3小时
    
2️⃣  模型2: rtdetr-mnv4-hybrid-m.yaml
    - 名称: RT-DETR+MobileNetV4
    - batch: 32, lr: 0.0004, epochs: 100
    - 预计显存: 20GB
    - 预计时间: 2-3小时

3️⃣  模型3: rtdetr-mnv4-hybrid-m-sea.yaml
    - 名称: RT-DETR+MobileNetV4+SEA
    - batch: 32, lr: 0.0004, epochs: 100
    - 预计显存: 20GB
    - 预计时间: 2-3小时

4️⃣  模型4: rtdetr-l-sea.yaml
    - 名称: RT-DETR-L+SEA
    - batch: 24, lr: 0.0004, epochs: 100
    - 预计显存: 21GB
    - 预计时间: 2-3小时

5️⃣  模型5: rtdetr-ghostnet.yaml
    - 名称: RT-DETR+GhostNet
    - batch: 48, lr: 0.0004, epochs: 100
    - 预计显存: 18GB
    - 预计时间: 1.5-2小时

6️⃣  模型6: rtdetr-shufflenet-sea.yaml
    - 名称: RT-DETR+ShuffleNet+SEA
    - batch: 48, lr: 0.0004, epochs: 100
    - 预计显存: 18GB
    - 预计时间: 1.5-2小时

7️⃣  模型7: rtdetr-efficientnet-cbam.yaml
    - 名称: RT-DETR+EfficientNet+CBAM
    - batch: 32, lr: 0.0004, epochs: 100
    - 预计显存: 19GB
    - 预计时间: 2-3小时

8️⃣  模型8: rtdetr-l-cbam.yaml
    - 名称: RT-DETR-L+CBAM
    - batch: 24, lr: 0.0004, epochs: 100
    - 预计显存: 21GB
    - 预计时间: 2-3小时

9️⃣  模型9: rtdetr-mobilenetv3.yaml
    - 名称: RT-DETR+MobileNetV3
    - batch: 56, lr: 0.0005, epochs: 120
    - 预计显存: 16GB
    - 预计时间: 1-1.5小时

🔟 模型10: rtdetr-repghostnet.yaml
    - 名称: RT-DETR+RepGhostNet
    - batch: 64, lr: 0.0005, epochs: 120
    - 预计显存: 15GB
    - 预计时间: 0.8-1.2小时

1️⃣1️⃣ 模型11: ert-detr.yaml (🔬 论文级轻量化创新模型)
    - 名称: ERT-DETR (Efficient Real-Time DETR)
    - batch: 48, lr: 0.0004, epochs: 150  
    - 预计显存: 14GB
    - 预计时间: 1-1.5小时
    - 🎯 创新点: 5大轻量化技术集成
      • MBConv + LightSEA + GroupedCBAM
      • AdaptiveChannelSelection + LinearAttention
EOF
}

# 显示使用方法
show_usage() {
    cat << 'EOF'
📖 使用方法:

显示所有可用模型:
    ./train.sh list

训练指定模型:
    ./train.sh <模型号>           # 例如: ./train.sh 1

训练多个模型 (顺序执行):
    ./train.sh <模型号1> <模型号2> ...  # 例如: ./train.sh 1 2 3

查看模型训练状态:
    ./train.sh status

停止当前训练:
    ./train.sh stop

显示训练日志:
    ./train.sh log <模型号>       # 例如: ./train.sh log 1

📝 示例:
    # 只训练模型1
    ./train.sh 1
    
    # 依次训练模型1、2、3
    ./train.sh 1 2 3
    
    # 查看模型1的训练进度
    tail -f training_model_1.log
EOF
}

# 检查模型号有效性
validate_model_id() {
    local id=$1
    if [[ ! "$id" =~ ^[1-9]$|^1[01]$ ]]; then
        echo "❌ 错误: 模型号必须在 1-11 之间"
        return 1
    fi
    return 0
}

# 检查是否已有训练进程在运行
check_existing_process() {
    local model_id=$1
    if pgrep -f "SELECTED_MODEL = '$model_id'" > /dev/null 2>&1; then
        return 0  # 进程存在
    fi
    if pgrep -f "training_model_${model_id}.log" > /dev/null 2>&1; then
        return 0  # 进程存在
    fi
    return 1  # 进程不存在
}

# 查看训练状态
show_status() {
    echo "🔍 检查训练进程..."
    echo ""
    
    local running_count=0
    for i in {1..11}; do
        if ps aux | grep -E "SELECTED_MODEL = '$i'|training_model_$i\.log" | grep -v grep > /dev/null 2>&1; then
            echo "✅ 模型 $i: 正在运行"
            ((running_count++))
            
            # 显示日志最后几行
            if [ -f training_model_${i}.log ]; then
                local last_line=$(tail -1 training_model_${i}.log)
                if [[ "$last_line" == *"Epoch"* ]] || [[ "$last_line" == *"epoch"* ]]; then
                    echo "   最后状态: $last_line"
                fi
            fi
        elif [ -f training_model_${i}.log ]; then
            # 检查是否已完成
            if grep -q "🎉 训练完成" training_model_${i}.log 2>/dev/null; then
                echo "✅ 模型 $i: 已完成"
            else
                echo "⏸️  模型 $i: 已停止"
            fi
        else
            echo "⭕ 模型 $i: 未启动"
        fi
    done
    
    echo ""
    echo "=========================================="
    if [ $running_count -eq 0 ]; then
        echo "📊 状态: 无进程在运行"
    else
        echo "📊 状态: $running_count 个进程正在运行"
    fi
    echo "=========================================="
}

# 停止所有训练
stop_training() {
    echo "🛑 停止所有训练进程..."
    pkill -f "auto_train_rtdetr" || true
    sleep 2
    
    if pgrep -f "auto_train_rtdetr" > /dev/null 2>&1; then
        echo "⚠️  强制停止..."
        pkill -9 -f "auto_train_rtdetr" || true
    fi
    
    echo "✅ 所有进程已停止"
}

# 显示日志
show_log() {
    local model_id=$1
    validate_model_id "$model_id" || return 1
    
    local log_file="training_model_${model_id}.log"
    if [ ! -f "$log_file" ]; then
        echo "❌ 日志文件不存在: $log_file"
        return 1
    fi
    
    echo "📝 显示模型 $model_id 的训练日志 (实时监控):"
    echo "💡 按 Ctrl+C 退出"
    echo ""
    tail -f "$log_file"
}

# 启动单个模型训练
start_training() {
    local model_id=$1
    
    # 验证模型号
    validate_model_id "$model_id" || return 1
    
    # 检查是否已运行
    if check_existing_process "$model_id"; then
        echo "⚠️  模型 $model_id 已在运行中，跳过..."
        return 0
    fi
    
    # 检查日志文件是否存在且未完成
    if [ -f "training_model_${model_id}.log" ]; then
        if ! grep -q "✅ 训练任务全部完成" "training_model_${model_id}.log" 2>/dev/null; then
            echo "⚠️  模型 $model_id 的日志已存在，正在覆盖..."
        fi
    fi
    
    echo "🚀 启动模型 $model_id 训练..."
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    
    # 修改脚本中的模型选择
    sed -i "s/^SELECTED_MODEL = '[0-9]'/SELECTED_MODEL = '$model_id'/" scripts/training/auto_train_rtdetr.py
    
    # 后台运行训练
    nohup python3 scripts/training/auto_train_rtdetr.py --skip-confirm > training_model_${model_id}.log 2>&1 &
    local train_pid=$!
    
    echo "✅ 模型 $model_id 已启动"
    echo "   📊 进程ID: $train_pid"
    echo "   📝 日志文件: training_model_${model_id}.log"
    echo "   🔍 实时查看: tail -f training_model_${model_id}.log"
    echo ""
    
    disown $train_pid
}

# 主程序
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    case "$1" in
        list)
            show_models
            ;;
        status)
            show_status
            ;;
        stop)
            stop_training
            ;;
        log)
            if [ -z "$2" ]; then
                echo "❌ 请指定模型号，例如: ./train.sh log 1"
                exit 1
            fi
            show_log "$2"
            ;;
        *)
            # 训练模式：可以接收一个或多个模型号
            echo "🎯 任务规划:"
            for model_id in "$@"; do
                if validate_model_id "$model_id"; then
                    echo "  - 模型 $model_id: 已加入队列"
                fi
            done
            echo ""
            
            # 顺序启动所有指定的模型
            for model_id in "$@"; do
                if validate_model_id "$model_id"; then
                    start_training "$model_id"
                    
                    # 检查是否需要等待（可选，取决于显存）
                    if [ -n "$WAIT_BETWEEN_MODELS" ]; then
                        echo "⏳ 等待模型 $model_id 完成..."
                        wait
                    fi
                fi
            done
            
            if [ $# -gt 0 ]; then
                echo "=========================================="
                echo "📊 训练已启动！"
                echo "📝 查看模型状态: ./train.sh status"
                echo "📖 查看日志: ./train.sh log <模型号>"
                echo "🛑 停止训练: ./train.sh stop"
                echo "=========================================="
            fi
            ;;
    esac
}

# 运行主程序
main "$@"
