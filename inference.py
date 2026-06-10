import argparse
import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.detector     import HelmetDetector, ANCHORS
from utils.nms           import post_process
from utils.segmentation  import draw_detections

CLASS_NAMES = ['full-faced', 'half-faced', 'invalid', 'no helmet']
IMG_SIZE    = 416


def preprocess(image_bgr, img_size=416):
    """OpenCV image → model input tensor"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=image_rgb)['image']
    return tensor.unsqueeze(0)   # (1, 3, H, W)


def run_inference(image_path, checkpoint_path, 
                  conf_thresh=0.4, iou_thresh=0.45,
                  draw_masks=True, save_output=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = HelmetDetector(num_classes=4).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"[checkpoint] Standard torch.load failed: {e}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            print("[checkpoint] Loaded with weights_only=False")
        except TypeError:
            try:
                import numpy as _np
                from torch.serialization import add_safe_globals
                add_safe_globals([_np._core.multiarray.scalar])
                ckpt = torch.load(checkpoint_path, map_location=device)
                print("[checkpoint] Loaded after adding safe globals for numpy.scalar")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load checkpoint '{checkpoint_path}': {e2}\n"
                    "If you trust the checkpoint, consider loading with torch.load(..., weights_only=False)"
                ) from e2

    model.load_state_dict(ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt)
    model.eval()
    print(f" Model loaded from {checkpoint_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    inp = preprocess(image_bgr, IMG_SIZE).to(device)

    with torch.no_grad():
        outputs = model(inp)

    boxes, scores, labels = post_process(
        outputs, ANCHORS,
        num_classes = 4,
        img_size    = IMG_SIZE,
        conf_thresh = conf_thresh,
        iou_thresh  = iou_thresh,
    )

    if len(boxes) > 0:
        boxes[:, [0, 2]] *= orig_w / IMG_SIZE
        boxes[:, [1, 3]] *= orig_h / IMG_SIZE

    print(f"\n Detected {len(boxes)} objects:")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"  [{i+1}] {CLASS_NAMES[int(label)]:15s} "
              f"conf={score:.3f}  "
              f"box=({int(box[0])},{int(box[1])}) → ({int(box[2])},{int(box[3])})")

    result_img = draw_detections(image_bgr, boxes, scores, labels,
                                 draw_masks=draw_masks)

    if save_output:
        os.makedirs('outputs', exist_ok=True)
        out_name = 'outputs/' + os.path.basename(image_path)
        cv2.imwrite(out_name, result_img)
        print(f"\n Saved: {out_name}")

    return result_img, boxes, scores, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', default='checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--conf',       type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--iou',        type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--no-masks',   action='store_true',
                        help='Skip segmentation masks (faster)')
    args = parser.parse_args()

    run_inference(
        image_path      = args.image,
        checkpoint_path = args.checkpoint,
        conf_thresh     = args.conf,
        iou_thresh      = args.iou,
        draw_masks      = not args.no_masks,
    )