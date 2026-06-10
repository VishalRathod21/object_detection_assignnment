import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import gc

from models.detector        import HelmetDetector, ANCHORS
from losses.detection_loss  import DetectionLoss
from utils.dataset          import HelmetDataset, collate_fn
from utils.metrics          import compute_map
from utils.nms              import post_process


CONFIG = {
    'train_images' : 'dataset/train/images',
    'train_labels' : 'dataset/train/labels',
    'val_images'   : 'dataset/valid/images',
    'val_labels'   : 'dataset/valid/labels',
    'num_classes'  : 4,
    'img_size'     : 416,
    'epochs'       : 3,
    'batch_size'   : 8,  # Reduced for 8GB RAM
    'grad_accum'   : 2,   # Accumulate 8 steps for effective batch size of 16
    'lr'           : 1e-3,
    'weight_decay' : 1e-4,
    'save_dir'     : 'checkpoints',
    'conf_thresh'  : 0.01,
    'iou_thresh'   : 0.45,
}


def evaluate(model, val_loader, device, epoch):
    """Validation loop — mAP calculate karta hai"""
    model.eval()
    all_preds = []
    all_gts   = []

    with torch.no_grad():
        for images, gt_boxes_list, gt_labels_list in val_loader:
            images = images.to(device)
            outputs = model(images)

            for i in range(len(images)):
                single_out = [o[i:i+1] for o in outputs]
                boxes, scores, labels = post_process(
                    single_out, ANCHORS,
                    num_classes  = CONFIG['num_classes'],
                    img_size     = CONFIG['img_size'],
                    conf_thresh  = CONFIG['conf_thresh'],
                    iou_thresh   = CONFIG['iou_thresh'],
                    model        = model,
                )
                all_preds.append({'boxes': boxes, 'scores': scores, 'labels': labels})

                gt_b = gt_boxes_list[i]
                gt_l = gt_labels_list[i]
                if len(gt_b) > 0:
                    cx  = gt_b[:, 0] * CONFIG['img_size']
                    cy  = gt_b[:, 1] * CONFIG['img_size']
                    w   = gt_b[:, 2] * CONFIG['img_size']
                    h   = gt_b[:, 3] * CONFIG['img_size']
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    gt_boxes_px = torch.stack([x1,y1,x2,y2], dim=-1)
                else:
                    gt_boxes_px = torch.zeros((0,4))
                all_gts.append({'boxes': gt_boxes_px, 'labels': gt_l})

    print(f"\n📊 Validation Results — Epoch {epoch+1}")
    mAP, aps = compute_map(all_preds, all_gts,
                           num_classes=CONFIG['num_classes'])
    print(f"   mAP@0.5 = {mAP:.4f}")
    return mAP


def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
    
    if device.type == 'cpu':
        torch.set_num_threads(4)
        print(f"🔧 CPU threads limited to 4 for memory efficiency")
    
    torch.backends.cudnn.benchmark = False

    train_ds = HelmetDataset(CONFIG['train_images'], CONFIG['train_labels'],
                             img_size=CONFIG['img_size'], is_train=True)
    val_ds   = HelmetDataset(CONFIG['val_images'],   CONFIG['val_labels'],
                             img_size=CONFIG['img_size'], is_train=False)

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=0,  # Single-process for CPU
                              collate_fn=collate_fn, pin_memory=use_cuda)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0,  # Single-process for CPU
                              collate_fn=collate_fn, pin_memory=use_cuda)

    model     = HelmetDetector(num_classes=CONFIG['num_classes']).to(device)
    criterion = DetectionLoss(num_classes=CONFIG['num_classes'], model=model)
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG['lr'],
                            weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-5)

    best_map = -1.0

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_obj  = 0.0
        epoch_cls  = 0.0
        epoch_box  = 0.0

        for step, (images, gt_boxes_list, gt_labels_list) in enumerate(train_loader):
            load_start = time.time()
            images = images.to(device)
            load_time = time.time() - load_start
            
            fwd_start = time.time()
            outputs = model(images)
            fwd_time = time.time() - fwd_start
            
            loss_start = time.time()
            loss, obj_l, cls_l, box_l = criterion(
                outputs, gt_boxes_list, gt_labels_list
            )
            loss_time = time.time() - loss_start
            
            loss = loss / CONFIG['grad_accum']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            if (step + 1) % CONFIG['grad_accum'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * CONFIG['grad_accum']
            epoch_obj  += obj_l
            epoch_cls  += cls_l
            epoch_box  += box_l

            if step % 20 == 0:
                if device.type == 'cpu':
                    mem_mb = 0
                else:
                    mem_mb = torch.cuda.memory_allocated(device) / 1024**2
                
                print(f"Ep[{epoch+1:03d}/{CONFIG['epochs']}] "
                      f"Step[{step:04d}/{len(train_loader):04d}] "
                      f"Loss={loss.item()*CONFIG['grad_accum']:.4f}  "
                      f"Obj={obj_l:.3f}  "
                      f"Cls={cls_l:.3f}  "
                      f"Box={box_l:.3f}  "
                      f"Load={load_time:.3f}s  "
                      f"Fwd={fwd_time:.3f}s  "
                      f"Loss={loss_time:.3f}s  "
                      f"Mem={mem_mb:.0f}MB")
                
                if step % 100 == 0:
                    gc.collect()

        if len(train_loader) % CONFIG['grad_accum'] != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n Epoch {epoch+1} | AvgLoss={avg_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 1 == 0:
            mAP = evaluate(model, val_loader, device, epoch)

            if mAP > best_map:
                best_map = mAP
                ckpt_path = os.path.join(CONFIG['save_dir'], 'best.pth')
                torch.save({
                    'epoch'      : epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                    'mAP'        : best_map,
                    'config'     : CONFIG,
                }, ckpt_path)
                print(f" Best model saved! mAP={best_map:.4f} → {ckpt_path}")

        torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mAP': mAP,
        'config': CONFIG,
}, os.path.join(CONFIG['save_dir'], f'epoch_{epoch+1}.pth'))

        print(f"Checkpoint saved: epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train()