import numpy as np
from utils.nms import box_iou
import torch


def compute_ap(recall, precision):
    """
    Area under Precision-Recall curve.
    11-point interpolation method.
    """
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = precision[recall >= thr]
        ap += (prec_at_rec.max() if len(prec_at_rec) > 0 else 0.0)
    return ap / 11.0


def compute_map(all_predictions, all_ground_truths,
                num_classes=4, iou_threshold=0.5):
    """
    all_predictions  : list of dicts
                       {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
    all_ground_truths: list of dicts
                       {'boxes': (M,4), 'labels': (M,)}
    
    Returns: mAP float, per-class AP dict
    """
    CLASS_NAMES = ['full-faced', 'half-faced', 'invalid', 'no helmet']
    aps = {}

    for cls in range(num_classes):
        tp_list     = []
        fp_list     = []
        score_list  = []
        n_gt        = 0

        for preds, gts in zip(all_predictions, all_ground_truths):
            pred_mask = preds['labels'] == cls
            gt_mask   = gts['labels']  == cls

            pred_boxes  = preds['boxes'][pred_mask]
            pred_scores = preds['scores'][pred_mask]
            gt_boxes    = gts['boxes'][gt_mask]

            n_gt       += len(gt_boxes)
            matched_gt  = set()

            if len(pred_boxes) == 0:
                continue

            order = pred_scores.argsort(descending=True)
            pred_boxes  = pred_boxes[order]
            pred_scores = pred_scores[order]

            for pb, ps in zip(pred_boxes, pred_scores):
                score_list.append(ps.item())

                if len(gt_boxes) == 0:
                    tp_list.append(0)
                    fp_list.append(1)
                    continue

                ious = box_iou(pb.unsqueeze(0), gt_boxes)  # (1, n_gt)
                best_iou_vals, best_j_idxs = ious.max(dim=1)  # over cols
                best_iou = float(best_iou_vals.item())
                best_j = int(best_j_idxs.item())

                if best_iou >= iou_threshold and best_j not in matched_gt:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched_gt.add(best_j)
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        if n_gt == 0:
            aps[CLASS_NAMES[cls]] = 0.0
            continue

        sorted_idx = np.argsort(-np.array(score_list))
        tp = np.array(tp_list)[sorted_idx]
        fp = np.array(fp_list)[sorted_idx]

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recall    = cum_tp / (n_gt + 1e-7)
        precision = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap = compute_ap(recall, precision)
        aps[CLASS_NAMES[cls]] = ap
        print(f"  {CLASS_NAMES[cls]:15s}: AP = {ap:.4f}  "
              f"(GT={n_gt}, TP={tp.sum()}, FP={fp.sum()})")

    mAP = np.mean(list(aps.values()))
    return mAP, aps