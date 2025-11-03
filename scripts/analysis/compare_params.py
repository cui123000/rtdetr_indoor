import sys
sys.path.insert(0, 'ultralytics')
from ultralytics import RTDETR

models = [
    ('SEA基线', 'ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml'),
    ('ASFF v1', 'ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-dysample.yaml'),
    ('ASFF v2', 'ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v2.yaml'),
    ('ASFF v3', 'ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-asff-v3.yaml'),
]

print('📊 参数量对比分析\n')
print(f"{'模型':<15} {'总参数':<12} {'vs SEA':<12} {'ASFF模块':<10}")
print('-' * 55)

sea_params = None
for name, path in models:
    model = RTDETR(path)
    total = sum(p.numel() for p in model.model.parameters())
    
    # 统计ASFF模块
    asff_count = sum(1 for _, m in model.model.named_modules() if 'ASFF' in type(m).__name__)
    
    if sea_params is None:
        sea_params = total
        diff = '-'
    else:
        diff = f'{(total - sea_params) / 1e6:+.2f}M'
    
    print(f'{name:<15} {total/1e6:>6.2f}M    {diff:<12} {asff_count} 个')
    
print('\n💡 分析:')
print(f'  • ASFF v1: 1个ASFF_Simple，性能差 (mAP50 0.3927)')
print(f'  • ASFF v2: 3个ASFF，256通道，RepC3×3')
print(f'  • ASFF v3: 3个ASFF，224通道，RepC3×2 (平衡版)')
print(f'\n📈 性能回顾:')
print(f'  • SEA基线: 29.06M, mAP50 0.4782 ⭐')
print(f'  • ASFF v1: 27.75M, mAP50 0.3927 ❌')
print(f'  • ASFF v2: 29.78M, 待训练 🔄')
print(f'  • ASFF v3: 待验证, 待训练 🔄')
