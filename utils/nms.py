"""
Non-Maximum Suppression — from scratch
Overlapping boxes mein se best wala rakhta hai
"""

import torch


def box_iou(box1, box2):
    """
    box1: (N,4) -> x1,y1,x2,y2
    box2: (M,4) -> x1,y1,x2,y2

    Returns:
        IoU matrix (N,M)
    """

    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * \
            (box1[:, 3] - box1[:, 1]).clamp(min=0)

    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * \
            (box2[:, 3] - box2[:, 1]).clamp(min=0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + 1e-7

    return inter / union


def nms(boxes, scores, iou_threshold=0.45):
    """
    boxes  : (N, 4) x1y1x2y2
    scores : (N,)
    Returns: selected indices list
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]

        ix1 = x1[rest].clamp(min=x1[i].item())
        iy1 = y1[rest].clamp(min=y1[i].item())
        ix2 = x2[rest].clamp(max=x2[i].item())
        iy2 = y2[rest].clamp(max=y2[i].item())

        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

        union = areas[i] + areas[rest] - inter + 1e-7
        iou = inter / union

        order = rest[iou < iou_threshold]

    return keep


def post_process(outputs, anchors_list, num_classes=4,
                 img_size=416, conf_thresh=0.4, iou_thresh=0.45, model=None):
    """
    Model outputs → final detections for ONE image
    """

    from models.detector import HelmetDetector

    all_boxes = []
    all_scores = []
    all_labels = []

    for raw_out, anchors in zip(outputs, anchors_list):

        if model is not None:
            decoded_boxes, pred_obj, pred_cls = model.decode(
                raw_out,
                anchors,
                img_size
            )
        else:
            decoded_boxes, pred_obj, pred_cls = HelmetDetector.decode(
                HelmetDetector(),
                raw_out,
                anchors,
                img_size
            )

        decoded_boxes = decoded_boxes[0]
        pred_obj = pred_obj[0]
        pred_cls = pred_cls[0]

        boxes = decoded_boxes.reshape(-1, 4)
        obj = pred_obj.reshape(-1)
        cls = pred_cls.reshape(-1, num_classes)

        cls_score, cls_label = cls.max(dim=-1)
        final_score = obj * cls_score

        keep_mask = final_score > conf_thresh

        if keep_mask.sum() == 0:
            continue

        all_boxes.append(boxes[keep_mask])
        all_scores.append(final_score[keep_mask])
        all_labels.append(cls_label[keep_mask])

    if len(all_boxes) == 0:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.long)
        )

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    final_boxes = []
    final_scores = []
    final_labels = []

    for cls_id in range(num_classes):

        cls_mask = all_labels == cls_id

        if cls_mask.sum() == 0:
            continue

        c_boxes = all_boxes[cls_mask]
        c_scores = all_scores[cls_mask]

        keep = nms(c_boxes, c_scores, iou_thresh)

        final_boxes.append(c_boxes[keep])
        final_scores.append(c_scores[keep])

        final_labels.append(
            torch.full(
                (len(keep),),
                cls_id,
                dtype=torch.long
            )
        )

    if len(final_boxes) == 0:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.long)
        )

    return (
        torch.cat(final_boxes),
        torch.cat(final_scores),
        torch.cat(final_labels)
    )