#!/usr/bin/env python
"""
RT-DETR 训练监控脚本
实时监控GPU使用、训练进度和系统状态
"""

import os
import time
import subprocess
import json
from pathlib import Path

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp, power = result.stdout.strip().split(',')
            return {
                'gpu_util': float(gpu_util),
                'mem_used': float(mem_used),
                'mem_total': float(mem_total),
                'temp': float(temp),
                'power': float(power)
            }
    except:
        pass
    return None

def get_training_progress(log_file):
    """从日志获取训练进度"""
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            logs = [json.loads(line) for line in f if line.strip()]
        
        if not logs:
            return None
        
        last_log = logs[-1]
        return {
            'epoch': last_log.get('epoch', 0) + 1,
            'total_epochs': 100,
            'train_loss': last_log.get('train_loss', 0),
            'mAP': last_log.get('test_coco_eval_bbox', [0])[0] if 'test_coco_eval_bbox' in last_log else 0
        }
    except:
        return None

def main():
    output_dir = Path('/home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k')
    log_file = output_dir / 'log.txt'
    
    print("\n" + "="*80)
    print("🔍 RT-DETR 训练监控")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file}")
    print("="*80 + "\n")
    
    while True:
        try:
            # 清屏
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("\n" + "="*80)
            print(f"⏰ 更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # GPU信息
            gpu_info = get_gpu_info()
            if gpu_info:
                print(f"\n📊 GPU 状态:")
                print(f"  利用率   : {gpu_info['gpu_util']:.1f}%")
                print(f"  显存使用 : {gpu_info['mem_used']:.0f}/{gpu_info['mem_total']:.0f} MB ({gpu_info['mem_used']/gpu_info['mem_total']*100:.1f}%)")
                print(f"  温度     : {gpu_info['temp']:.1f}°C")
                print(f"  功耗     : {gpu_info['power']:.1f}W")
                
                # 警告检查
                if gpu_info['mem_used'] / gpu_info['mem_total'] > 0.9:
                    print(f"  ⚠️  警告: GPU显存使用过高!")
                if gpu_info['temp'] > 85:
                    print(f"  ⚠️  警告: GPU温度过高!")
            else:
                print("\n❌ 无法获取GPU信息")
            
            # 训练进度
            progress = get_training_progress(log_file)
            if progress:
                print(f"\n📈 训练进度:")
                print(f"  Epoch    : {progress['epoch']}/{progress['total_epochs']} ({progress['epoch']/progress['total_epochs']*100:.1f}%)")
                print(f"  训练损失 : {progress['train_loss']:.4f}")
                print(f"  mAP      : {progress['mAP']:.4f}")
            else:
                print("\n⏳ 等待训练启动...")
            
            print("\n" + "="*80)
            print("按 Ctrl+C 退出监控")
            print("="*80 + "\n")
            
            # 每5秒更新一次
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            time.sleep(5)

if __name__ == '__main__':
    main()
