"""
对比三个模式的训练结果并进行消融实验。
"""

import pandas as pd
import os
import plotly.express as px

# 定义文件路径
file_paths = {
    'rtdetr_l': '/root/autodl-tmp/runs/detect/rtdetr_l_rtx3080ti/results.csv',
    'rtdetr_mnv4_hybrid': '/root/autodl-tmp/runs/detect/rtdetr_mnv4_hybrid_rtx3080ti/results.csv',
    'rtdetr_mnv4_sea': '/root/autodl-tmp/runs/detect/rtdetr_mnv4_sea_rtx3080ti/results.csv'
}

# 加载数据
data = {}
for mode, path in file_paths.items():
    if os.path.exists(path):
        data[mode] = pd.read_csv(path)
    else:
        print(f"文件 {path} 不存在！")

# 更新输出目录为 rtdetr_indoor/output_images
output_base_dir = os.path.join(os.getcwd(), "output_images")
os.makedirs(output_base_dir, exist_ok=True)

# 确保每种模式的子目录存在
def ensure_mode_directory(mode):
    mode_dir = os.path.join(output_base_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    return mode_dir

# 绘制训练损失曲线
def plot_loss_interactive(data, mode, loss_type):
    mode_dir = ensure_mode_directory(mode)
    fig = px.line(data, x='epoch', y=loss_type, title=f"{mode} - {loss_type}", labels={'epoch': 'Epoch', loss_type: 'Loss'})
    # 修正路径生成逻辑，避免多余的路径分隔符
    sanitized_loss_type = loss_type.replace('/', '_')  # 替换非法字符
    fig.write_html(os.path.join(mode_dir, f"{sanitized_loss_type}_loss.html"))

for mode, df in data.items():
    for loss_type in ['train/giou_loss', 'train/cls_loss', 'train/l1_loss']:
        plot_loss_interactive(df, mode, loss_type)

# 绘制验证指标曲线
def plot_metrics_interactive(data, mode, metric):
    mode_dir = ensure_mode_directory(mode)
    fig = px.line(data, x='epoch', y=metric, title=f"{mode} - {metric}", labels={'epoch': 'Epoch', metric: 'Metric'})
    # 修正路径生成逻辑，避免多余的路径分隔符
    sanitized_metric = metric.replace('/', '_')  # 替换非法字符
    fig.write_html(os.path.join(mode_dir, f"{sanitized_metric}_metric.html"))

metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
for metric in metrics:
    for mode, df in data.items():
        plot_metrics_interactive(df, mode, metric)

# 输出最终性能对比
final_metrics = {}
for mode, df in data.items():
    final_metrics[mode] = {
        metric: df[metric].iloc[-1] for metric in metrics
    }

final_metrics_df = pd.DataFrame(final_metrics).T
print("最终性能对比：")
print(final_metrics_df)
final_metrics_df.to_csv("output_images/final_metrics_comparison.csv")