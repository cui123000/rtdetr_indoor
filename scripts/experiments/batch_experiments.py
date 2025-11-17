#!/usr/bin/env python3
"""
批量对比实验脚本
自动训练多个模型并生成对比报告
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 实验配置
EXPERIMENTS = {
    'baseline': {
        'model': 'rtdetr-l.yaml',
        'name': 'rtdetr_l_baseline',
        'batch': 12,
        'epochs': 50,  # 快速对比用50 epochs
        'lr0': 0.0001,
        'description': 'RT-DETR-L 基线模型'
    },
    'sea': {
        'model': 'rtdetr-l-sea.yaml',
        'name': 'rtdetr_l_sea',
        'batch': 10,
        'epochs': 50,
        'lr0': 0.0001,
        'description': 'RT-DETR-L + SEA注意力'
    },
    'cbam': {
        'model': 'rtdetr-l-cbam.yaml',
        'name': 'rtdetr_l_cbam',
        'batch': 10,
        'epochs': 50,
        'lr0': 0.0001,
        'description': 'RT-DETR-L + CBAM注意力'
    },
    'strong_aug': {
        'model': 'rtdetr-l.yaml',
        'name': 'rtdetr_l_strong_aug',
        'batch': 12,
        'epochs': 50,
        'lr0': 0.0001,
        'mosaic': 0.8,
        'mixup': 0.2,
        'copy_paste': 0.15,
        'description': 'RT-DETR-L + 强数据增强'
    },
}

def create_experiment_config(exp_name, exp_config):
    """为每个实验创建配置"""
    config = {
        'model': f'/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{exp_config["model"]}',
        'data': '/home/cjj/rtdetr_indoor/datasets/coco_indoor/coco_indoor.yaml',
        'epochs': exp_config['epochs'],
        'batch': exp_config['batch'],
        'lr0': exp_config['lr0'],
        'device': '0',
        'workers': 6,
        'project': '/home/cjj/rtdetr_indoor/runs/experiments',
        'name': f'{exp_config["name"]}_{datetime.now().strftime("%Y%m%d")}',
        'exist_ok': True,
    }
    
    # 特殊配置
    if 'mosaic' in exp_config:
        config['mosaic'] = exp_config['mosaic']
    if 'mixup' in exp_config:
        config['mixup'] = exp_config['mixup']
    if 'copy_paste' in exp_config:
        config['copy_paste'] = exp_config['copy_paste']
    
    return config

def run_experiment(exp_name, exp_config):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"🧪 实验: {exp_name}")
    print(f"📝 描述: {exp_config['description']}")
    print(f"{'='*70}\n")
    
    # 导入 RTDETR
    sys.path.insert(0, '/home/cjj/rtdetr_indoor')
    sys.path.insert(0, '/home/cjj/rtdetr_indoor/ultralytics')
    from ultralytics import RTDETR
    
    # 创建配置
    config = create_experiment_config(exp_name, exp_config)
    
    # 开始训练
    try:
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = time.time()
        
        model = RTDETR(config['model'])
        results = model.train(**{k: v for k, v in config.items() if k != 'model'})
        
        train_time = (time.time() - start_time) / 3600
        print(f"✅ 实验完成，用时: {train_time:.2f} 小时")
        
        # 保存实验记录
        record = {
            'name': exp_name,
            'description': exp_config['description'],
            'config': exp_config,
            'train_time_hours': train_time,
            'timestamp': datetime.now().isoformat(),
            'save_dir': config['project'] + '/' + config['name']
        }
        
        record_file = Path(config['project']) / f"{exp_name}_record.json"
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)
        
        return True, record
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}

def generate_comparison_report(results):
    """生成对比报告"""
    report_path = '/home/cjj/rtdetr_indoor/runs/experiments/comparison_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# RT-DETR 实验对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验列表\n\n")
        f.write("| 实验名称 | 描述 | 训练时间 | 状态 |\n")
        f.write("|---------|------|---------|------|\n")
        
        for exp_name, (success, record) in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            train_time = f"{record.get('train_time_hours', 0):.2f}h" if success else "N/A"
            desc = record.get('description', 'N/A')
            f.write(f"| {exp_name} | {desc} | {train_time} | {status} |\n")
        
        f.write("\n## 详细结果\n\n")
        for exp_name, (success, record) in results.items():
            if success:
                f.write(f"### {exp_name}\n\n")
                f.write(f"- **描述**: {record['description']}\n")
                f.write(f"- **训练时间**: {record['train_time_hours']:.2f} 小时\n")
                f.write(f"- **权重目录**: {record['save_dir']}\n")
                f.write(f"- **配置**: \n```json\n{json.dumps(record['config'], indent=2)}\n```\n\n")
    
    print(f"\n📊 对比报告已生成: {report_path}")

def main():
    """主函数"""
    print("🚀 批量实验开始")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧪 计划实验数: {len(EXPERIMENTS)}")
    
    # 用户确认
    print("\n实验列表:")
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"  - {exp_name}: {exp_config['description']}")
    
    confirm = input("\n确认开始所有实验? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消")
        return
    
    # 运行所有实验
    results = {}
    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items(), 1):
        print(f"\n📍 进度: {i}/{len(EXPERIMENTS)}")
        success, record = run_experiment(exp_name, exp_config)
        results[exp_name] = (success, record)
        
        # 保存中间结果
        if success:
            print(f"✅ {exp_name} 完成")
        else:
            print(f"❌ {exp_name} 失败")
    
    # 生成报告
    generate_comparison_report(results)
    
    # 总结
    print(f"\n{'='*70}")
    print("🎉 所有实验完成")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = sum(1 for s, _ in results.values() if s)
    print(f"✅ 成功: {success_count}/{len(EXPERIMENTS)}")
    print(f"❌ 失败: {len(EXPERIMENTS) - success_count}/{len(EXPERIMENTS)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
