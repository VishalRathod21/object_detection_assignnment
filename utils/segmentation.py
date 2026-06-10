import cv2
import numpy as np


CLASS_COLORS = {
    0: (0,   200,  0),    # full-faced  → Green
    1: (255, 165,  0),    # half-faced  → Orange
    2: (255,   0,  0),    # invalid     → Red
    3: (128,   0, 128),   # no helmet   → Purple
}

CLASS_NAMES = ['full-faced', 'half-faced', 'invalid', 'no helmet']


def get_segmentation_mask(image_bgr, box_x1y1x2y2, method='grabcut'):
    
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box_x1y1x2y2)

    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(W, x2);  y2 = min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W), dtype=np.uint8)

    if method == 'rect':
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    elif method == 'grabcut':
        mask = np.zeros((H, W), dtype=np.uint8)
        rect = (x1, y1, x2 - x1, y2 - y1)

        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(image_bgr, mask, rect,
                        bgd, fgd, iterCount=5,
                        mode=cv2.GC_INIT_WITH_RECT)
            result = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
        except Exception:
            result = np.zeros((H, W), dtype=np.uint8)
            result[y1:y2, x1:x2] = 1

        return result


def calculate_area(mask):
    """
    Pixel area = mask mein 1 wale pixels ka count
    """
    return int(mask.sum())


def draw_detections(image_bgr, boxes, scores, labels,
                    class_names=CLASS_NAMES, draw_masks=True):
    """
    Image pe boxes, labels, masks draw karo.
    
    Returns: annotated image (copy)
    """
    result = image_bgr.copy()

    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        color = CLASS_COLORS.get(label, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)

        if draw_masks:
            mask = get_segmentation_mask(result, [x1, y1, x2, y2],
                                         method='grabcut')
            area = calculate_area(mask)

            overlay        = result.copy()
            overlay[mask == 1] = color
            result         = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        else:
            area = (x2 - x1) * (y2 - y1)   # bbox area fallback

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        cls_name = class_names[label] if label < len(class_names) else str(label)
        label_text = f"{cls_name} {score:.2f} | {area}px"

        (tw, th), _ = cv2.getTextSize(label_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result,
                      (x1, y1 - th - 8), (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(result, label_text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return result