#!/usr/bin/env python3
"""
检查 YOLO 格式标签的有效性：class, bbox 归一化范围, w/h > 0, 无 NaN/Inf。
用法: python scripts/debug/check_dataset.py --data /path/to/dataset.yaml
如果不传 --data，会读取默认配置文件路径写在脚本顶部（可修改）。
"""
import argparse
import os
import glob
import math
import yaml
from pathlib import Path


def is_finite(x):
    return not (math.isinf(x) or math.isnan(x))


def check_label_file(file_path, nc=None):
    bad = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                bad.append((i+1, 'format', line))
                continue
            try:
                cls = int(float(parts[0]))
                vals = list(map(float, parts[1:5]))
            except Exception as e:
                bad.append((i+1, f'parse_err:{e}', line))
                continue
            x, y, w, h = vals
            if nc is not None and (cls < 0 or cls >= nc):
                bad.append((i+1, f'class_out_of_range:{cls}', line))
            for vname, v in (('x', x), ('y', y), ('w', w), ('h', h)):
                if not is_finite(v):
                    bad.append((i+1, f'{vname}_not_finite', line))
                if vname in ('x', 'y', 'w', 'h') and (v < 0.0 or v > 1.0):
                    bad.append((i+1, f'{vname}_out_of_0_1:{v}', line))
            if w <= 0.0 or h <= 0.0:
                bad.append((i+1, f'zero_or_negative_wh:{w},{h}', line))
    return bad


def find_label_files_from_yaml(yaml_path):
    root = Path(yaml_path).parent
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # expect data['train'] to be path (str) or list
    train = data.get('train')
    nc = data.get('nc')
    if isinstance(train, str):
        # if train is a glob or a directory or a file list
        if os.path.isdir(train):
            # labels in labels/train or labels
            candidates = [os.path.join(train, 'labels', 'train'), os.path.join(train, 'labels')]
            label_files = []
            for c in candidates:
                if os.path.isdir(c):
                    label_files.extend(glob.glob(os.path.join(c, '**', '*.txt'), recursive=True))
            if not label_files:
                # try images folder and replace images extension
                imgs = glob.glob(os.path.join(train, '**', '*.*'), recursive=True)
                label_files = [os.path.splitext(p)[0]+'.txt' for p in imgs]
                label_files = [p for p in label_files if os.path.exists(p)]
            return sorted(set(label_files)), nc
        elif os.path.isfile(train):
            # train may be a txt listing image paths
            label_files = []
            with open(train, 'r', encoding='utf-8') as f:
                for l in f:
                    l = l.strip()
                    if not l:
                        continue
                    lab = os.path.splitext(l)[0] + '.txt'
                    if os.path.exists(lab):
                        label_files.append(lab)
            return sorted(set(label_files)), nc
        else:
            # fallback: treat as path relative to yaml
            p = (root / train)
            if p.exists():
                # if relative path points to a directory, search for label files under it
                if p.is_dir():
                    candidates = [str(p / 'labels' / 'train'), str(p / 'labels')]
                    label_files = []
                    for c in candidates:
                        if os.path.isdir(c):
                            label_files.extend(glob.glob(os.path.join(c, '**', '*.txt'), recursive=True))
                    if not label_files:
                        imgs = glob.glob(os.path.join(str(p), '**', '*.*'), recursive=True)
                        label_files = [os.path.splitext(q)[0] + '.txt' for q in imgs]
                        label_files = [q for q in label_files if os.path.exists(q)]
                    return sorted(set(label_files)), nc
                elif p.is_file():
                    return find_label_files_from_yaml(str(p))
    elif isinstance(train, (list, tuple)):
        label_files = []
        for item in train:
            if os.path.exists(item):
                if os.path.isdir(item):
                    label_files.extend(glob.glob(os.path.join(item, '**', '*.txt'), recursive=True))
                elif os.path.isfile(item):
                    label_files.append(item)
        return sorted(set(label_files)), nc
    # as a last resort search for labels folder in yaml dir
    candidates = glob.glob(os.path.join(root, '**', 'labels', '**', '*.txt'), recursive=True)
    return sorted(set(candidates)), nc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='path to dataset yaml')
    parser.add_argument('--limit', type=int, default=None, help='max number of label files to check')
    parser.add_argument('--out', type=str, default='runs/debug/bad_labels.txt', help='output file for bad labels')
    args = parser.parse_args()

    yaml_path = args.data
    if yaml_path is None:
        # try default location
        yaml_path = 'datasets/homeobjects-3K/HomeObjects-3K.yaml'
        if not os.path.exists(yaml_path):
            print('请通过 --data 指定 dataset yaml 路径')
            return
    label_files, nc = find_label_files_from_yaml(yaml_path)
    print(f'Found {len(label_files)} label files, nc={nc}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bad_total = 0
    with open(args.out, 'w', encoding='utf-8') as out_f:
        for i, lf in enumerate(label_files):
            if args.limit and i >= args.limit:
                break
            bad = check_label_file(lf, nc)
            if bad:
                bad_total += 1
                out_f.write(f'FILE: {lf}\n')
                for b in bad:
                    out_f.write(f'  line {b[0]}: {b[1]} -> {b[2]}\n')
                out_f.write('\n')
    print(f'Checked {min(len(label_files), args.limit or len(label_files))} files. Bad files: {bad_total}. Saved to {args.out}')

if __name__ == '__main__':
    main()
