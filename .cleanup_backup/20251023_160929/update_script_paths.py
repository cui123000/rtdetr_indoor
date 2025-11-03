#!/usr/bin/env python3
"""
更新脚本路径引用
批量修复文件移动后的路径问题
"""

import os
import re
from pathlib import Path

def update_file_references():
    """更新文件中的路径引用"""
    
    # 需要更新的文件映射
    path_updates = {
        'train_rtdetr_mobilenetv4_select.py': 'scripts/training/train_rtdetr_mobilenetv4_select.py',
        'train_rtdetr_mobilenetv4.py': 'scripts/training/train_rtdetr_mobilenetv4.py',
        'train_sea_lite_optimized.py': 'scripts/training/train_sea_lite_optimized.py',
        'test_hybrid_model.py': 'scripts/evaluation/test_hybrid_model.py',
        'test_all_versions.py': 'scripts/evaluation/test_all_versions.py',
        'diagnose_sea_performance.py': 'scripts/analysis/diagnose_sea_performance.py',
        'compare_configurations.py': 'scripts/analysis/compare_configurations.py',
        'visualize_fusion_architecture.py': 'scripts/analysis/visualize_fusion_architecture.py',
        'fusion_implementation_summary.py': 'scripts/analysis/fusion_implementation_summary.py',
        'ablation_study.py': 'scripts/ablation/ablation_study.py',
        'run_ablation_experiments.py': 'scripts/ablation/run_ablation_experiments.py',
        'analyze_ablation_results.py': 'scripts/ablation/analyze_ablation_results.py',
        'check_ablation_environment.py': 'scripts/ablation/check_ablation_environment.py',
        'quick_start_optimization.py': 'scripts/utils/quick_start_optimization.py',
        'optimize_sea_training.py': 'scripts/utils/optimize_sea_training.py'
    }
    
    # 需要检查的目录
    directories = [
        '/home/cui/rtdetr_indoor/scripts',
        '/home/cui/rtdetr_indoor/docs'
    ]    # 需要检查的文件类型
    file_extensions = ['.py', '.md', '.yaml', '.yml', '.sh']
    
    updated_files = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if not any(file.endswith(ext) for ext in file_extensions):
                    continue
                    
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # 更新路径引用
                    for old_path, new_path in path_updates.items():
                        # 更新python调用
                        pattern1 = rf'python\s+{re.escape(old_path)}'
                        replacement1 = f'python {new_path}'
                        content = re.sub(pattern1, replacement1, content)
                        
                        # 更新import引用
                        pattern2 = rf'from\s+{re.escape(old_path.replace(".py", ""))}\s+import'
                        replacement2 = f'from {new_path.replace(".py", "").replace("/", ".")} import'
                        content = re.sub(pattern2, replacement2, content)
                        
                        # 更新直接文件引用
                        pattern3 = rf'["\']' + re.escape(old_path) + r'["\']'
                        replacement3 = f'"{new_path}"'
                        content = re.sub(pattern3, replacement3, content)
                    
                    # 如果内容有变化，写回文件
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        updated_files.append(filepath)
                        print(f"✅ 更新: {filepath}")
                        
                except Exception as e:
                    print(f"❌ 错误处理文件 {filepath}: {e}")
    
    return updated_files

def update_guide_files():
    """更新指南文件中的路径"""
    additional_files = [
        '/home/cui/rtdetr_indoor/ABLATION_GUIDE.md',
        '/home/cui/rtdetr_indoor/READY_TO_START.md'
    ]
    
    path_updates = {
        'python run_ablation_experiments.py': 'python scripts/ablation/run_ablation_experiments.py',
        'python check_ablation_environment.py': 'python scripts/ablation/check_ablation_environment.py',
        'python analyze_ablation_results.py': 'python scripts/ablation/analyze_ablation_results.py',
        'python train_rtdetr_mobilenetv4_select.py': 'python scripts/training/train_rtdetr_mobilenetv4_select.py'
    }
    
    for guide_file in guide_files:
        if not os.path.exists(guide_file):
            continue
            
        try:
            with open(guide_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            for old_cmd, new_cmd in path_updates.items():
                content = content.replace(old_cmd, new_cmd)
            
            if content != original_content:
                with open(guide_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 更新指南: {guide_file}")
                
        except Exception as e:
            print(f"❌ 错误处理指南文件 {guide_file}: {e}")

def main():
    """主函数"""
    print("🔄 更新脚本路径引用")
    print("=" * 50)
    
    print("\n📝 更新脚本文件...")
    updated_files = update_file_references()
    
    print(f"\n📖 更新指南文件...")
    update_guide_files()
    
    print(f"\n✅ 完成!")
    print(f"📊 总共更新了 {len(updated_files)} 个文件")
    
    if updated_files:
        print(f"\n📋 更新的文件:")
        for file in updated_files:
            print(f"  - {file}")

if __name__ == "__main__":
    main()
