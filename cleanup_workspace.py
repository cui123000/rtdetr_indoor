#!/usr/bin/env python3
"""
工作区清理脚本
清理过时和多余的文件，整理目录结构
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# 工作区根目录
ROOT = Path("/home/cui/rtdetr_indoor")

# 要删除的文件和目录
TO_DELETE = [
    # 1. 根目录多余文件
    "clash-linux-amd64-2023.08.17-11-g0f901d0.gz",  # clash代理工具，与项目无关
    "yolo11n.pt",  # YOLO11权重，项目用RT-DETR
    "nohup.out",  # 临时日志文件
    "update_script_paths.py",  # 一次性使用的更新脚本
    
    # 2. 过时的训练脚本（已迁移到scripts/training/）
    "train_asff_quick.sh",  # 已过时，用train_mnv4_variants.py代替
    
    # 3. 临时输出目录
    "output_images/",  # 临时测试输出
]

# 要移动/重命名的文件
TO_ORGANIZE = {
    # 文档整理
    "RT-DETR_MobileNetV4_融合架构详解.md": "docs/RT-DETR_MobileNetV4_融合架构详解.md",
    "ABLATION_GUIDE.md": "docs/ABLATION_GUIDE.md",
    "READY_TO_START.md": "docs/READY_TO_START.md",
    
    # 过时的脚本（移到scripts/deprecated/）
    "scripts/train_rtdetr_sea.py": "scripts/deprecated/train_rtdetr_sea.py",
    "scripts/train_rtdetr_mnv4_sea.py": "scripts/deprecated/train_rtdetr_mnv4_sea.py",
    "scripts/simple_sea_test.py": "scripts/deprecated/simple_sea_test.py",
    "scripts/test_sea_attention.py": "scripts/deprecated/test_sea_attention.py",
    "scripts/test_optimized_sea.py": "scripts/deprecated/test_optimized_sea.py",
    "scripts/validate_model.py": "scripts/deprecated/validate_model.py",
}

# 新训练脚本（创建快捷方式）
TRAINING_SHORTCUTS = {
    "train_v3.sh": """#!/bin/bash
# 快捷训练脚本：ASFF v3 (推荐)
cd /home/cui/rtdetr_indoor
conda run -n uRTDETR python scripts/training/train_mnv4_variants.py --variant rtdetr_mnv4_sea_asff_v3
""",
    "train_v2.sh": """#!/bin/bash
# 快捷训练脚本：ASFF v2 (完整版)
cd /home/cui/rtdetr_indoor
conda run -n uRTDETR python scripts/training/train_mnv4_variants.py --variant rtdetr_mnv4_sea_asff_v2
""",
}


def backup_before_delete(path: Path) -> None:
    """备份要删除的文件"""
    if not path.exists():
        return
    
    backup_dir = ROOT / ".cleanup_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    if path.is_file():
        shutil.copy2(path, backup_dir / path.name)
        print(f"  📦 已备份: {path.name} → {backup_dir}")
    elif path.is_dir():
        shutil.copytree(path, backup_dir / path.name, dirs_exist_ok=True)
        print(f"  📦 已备份: {path.name}/ → {backup_dir}")


def main():
    print("=" * 80)
    print("🧹 RT-DETR 工作区清理")
    print("=" * 80)
    print()
    
    # ============ 步骤1: 删除多余文件 ============
    print("📋 步骤1: 删除多余文件")
    print("-" * 80)
    
    deleted_count = 0
    for item in TO_DELETE:
        path = ROOT / item
        if path.exists():
            # 备份
            backup_before_delete(path)
            
            # 删除
            if path.is_file():
                path.unlink()
                print(f"  ✅ 已删除文件: {item}")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"  ✅ 已删除目录: {item}")
            deleted_count += 1
        else:
            print(f"  ⏭️  跳过（不存在）: {item}")
    
    print(f"\n✨ 删除了 {deleted_count} 个项目\n")
    
    # ============ 步骤2: 整理文件位置 ============
    print("📋 步骤2: 整理文件位置")
    print("-" * 80)
    
    organized_count = 0
    for src, dst in TO_ORGANIZE.items():
        src_path = ROOT / src
        dst_path = ROOT / dst
        
        if not src_path.exists():
            print(f"  ⏭️  跳过（不存在）: {src}")
            continue
        
        # 创建目标目录
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 移动文件
        if dst_path.exists():
            print(f"  ⚠️  目标已存在，跳过: {src} → {dst}")
        else:
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ 已移动: {src} → {dst}")
            organized_count += 1
    
    print(f"\n✨ 整理了 {organized_count} 个文件\n")
    
    # ============ 步骤3: 创建训练快捷脚本 ============
    print("📋 步骤3: 创建训练快捷脚本")
    print("-" * 80)
    
    for script_name, content in TRAINING_SHORTCUTS.items():
        script_path = ROOT / script_name
        
        if script_path.exists():
            print(f"  ⏭️  跳过（已存在）: {script_name}")
        else:
            script_path.write_text(content)
            script_path.chmod(0o755)
            print(f"  ✅ 已创建: {script_name}")
    
    print()
    
    # ============ 步骤4: 清理空目录 ============
    print("📋 步骤4: 清理空目录")
    print("-" * 80)
    
    empty_dirs = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if not dirnames and not filenames and dirpath != str(ROOT):
            path = Path(dirpath)
            if path.name not in ['.git', '.idea', '__pycache__', '.cleanup_backup']:
                empty_dirs.append(path)
    
    for empty_dir in empty_dirs:
        try:
            empty_dir.rmdir()
            print(f"  ✅ 已删除空目录: {empty_dir.relative_to(ROOT)}")
        except OSError:
            pass
    
    if not empty_dirs:
        print("  ✨ 没有发现空目录")
    
    print()
    
    # ============ 总结 ============
    print("=" * 80)
    print("✅ 清理完成！")
    print("=" * 80)
    print()
    print("📁 工作区结构:")
    print("  ├─ docs/                    # 文档（已整理）")
    print("  ├─ scripts/")
    print("  │  ├─ training/            # 训练脚本（主要）")
    print("  │  ├─ analysis/            # 分析脚本")
    print("  │  ├─ evaluation/          # 评估脚本")
    print("  │  └─ deprecated/          # 过时脚本（已归档）")
    print("  ├─ ultralytics/            # Ultralytics框架")
    print("  ├─ datasets/               # 数据集")
    print("  ├─ runs/                   # 训练结果")
    print("  ├─ train_v3.sh            # 快捷训练脚本（v3推荐）⭐")
    print("  ├─ train_v2.sh            # 快捷训练脚本（v2完整版）")
    print("  └─ README.md              # 项目说明")
    print()
    print("🚀 快速开始:")
    print("  训练ASFF v3: bash train_v3.sh")
    print("  训练ASFF v2: bash train_v2.sh")
    print("  分析结果:     python scripts/analysis/compare_asff_versions.py")
    print()
    print("💾 备份位置: .cleanup_backup/")
    print()


if __name__ == "__main__":
    main()
