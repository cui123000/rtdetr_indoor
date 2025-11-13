import time
import torch
import sys
import os

# 添加正确的模块路径
sys.path.insert(0, '/home/cui/rtdetr_indoor/ultralytics')
os.chdir('/home/cui/rtdetr_indoor')

from ultralytics import RTDETR

models = {
    'RT-DETR-L': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml',
    'RT-DETR-MNV4': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml',
    'RT-DETR-MNV4-SEA': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml'
}

def count_params(module):
    return sum(p.numel() for p in module.parameters())

for name, cfg in models.items():
    print('---', name, '---')
    try:
        model = RTDETR(cfg)
    except Exception as e:
        print('Failed to create model from', cfg, 'error:', e)
        continue
    # try to find torch nn.Module inside
    net = None
    for attr in ['model','net','module','model.model']:
        if hasattr(model, attr):
            net = getattr(model, attr)
            break
    # if direct attr not found, try to inspect model
    if net is None:
        # attempt common attribute
        if hasattr(model, 'model'):
            net = model.model
    if net is None:
        # fallback: model itself may be nn.Module
        if hasattr(model, 'parameters'):
            net = model
    if net is None:
        print('Could not find nn.Module for', name)
        continue
    params = count_params(net)
    print('Parameters:', params, '({:.2f}M)'.format(params/1e6))
    # forward timing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy = torch.randn(1,3,640,640).to(device)
    net.to(device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    # warmup
    for _ in range(3):
        with torch.no_grad():
            _ = net(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    iters = 10
    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = net(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    avg_ms = (t1 - t0) / iters * 1000
    print('Avg forward time: {:.2f} ms'.format(avg_ms))
    # cleanup
    del net, model
    torch.cuda.empty_cache()
    time.sleep(1)
