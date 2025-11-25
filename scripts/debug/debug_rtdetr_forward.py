"""Debug helper: single-batch forward + loss for RTDETR to reproduce NaN.
Run: python scripts/debug/debug_rtdetr_forward.py
"""
import sys
from pathlib import Path
import torch

# ensure repo root on path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from ultralytics.nn.tasks import RTDETRDetectionModel

def main(device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # model yaml path (use ert-detr.yaml from repo)
    cfg_path = str(repo_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "rt-detr" / "ert-detr.yaml")
    print("Model cfg:", cfg_path)

    # create model
    model = RTDETRDetectionModel(cfg=cfg_path, ch=3, nc=12, verbose=False)
    model.to(device)
    model.train()

    # build synthetic single-image batch with one gt box
    B = 1
    imgsz = 640
    img = torch.rand(B, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    # normalize in [0,1] already

    # one GT box (xywh normalized) and class 0
    gt_bboxes = torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float32, device=device)  # (N,4)
    gt_cls = torch.tensor([0], dtype=torch.long, device=device)  # (N,)
    batch_idx = torch.tensor([0], dtype=torch.long, device=device)  # (N,)
    gt_groups = [1]

    batch = {
        "img": img,
        "cls": gt_cls,
        "bboxes": gt_bboxes,
        "batch_idx": batch_idx,
        "gt_groups": gt_groups,
    }

    # run forward/loss
    try:
        print("Running model(batch) ...")
        loss_sum, loss_items = model(batch)
        print("loss_sum:", loss_sum)
        print("loss_items:", loss_items)

        # inspect individual loss components by calling criterion directly
        if hasattr(model, 'criterion') and model.criterion is not None:
            print("Computing criterion(preds, targets) for more detail...")
            # preds from model.predict
            img = batch['img']
            bs = len(img)
            batch_idx = batch['batch_idx']
            gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
            targets = {
                'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
                'bboxes': batch['bboxes'].to(device=img.device),
                'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
                'gt_groups': gt_groups,
            }
            preds = model.predict(img, batch=targets)
            print('Top-level preds types:', type(preds))
            try:
                if model.training:
                    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds
                    print('dec_bboxes dtype/shape:', dec_bboxes.dtype, dec_bboxes.shape)
                else:
                    print('Model not in training mode while expecting training preds')
            except Exception as e:
                print('Error unpacking preds:', e)

            try:
                losses = model.criterion((dec_bboxes, dec_scores), targets)
                print('loss keys:', list(losses.keys()))
                for k, v in losses.items():
                    try:
                        print(f"{k}: min={v.detach().min().item()}, max={v.detach().max().item()}, is_nan={torch.isnan(v).any().item()}")
                    except Exception:
                        print(f"{k}: {v}")
            except Exception as e:
                print('Error computing criterion:', e)

    except Exception as e:
        print('Exception during forward/loss:', e)
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
