import torch
import torch.nn as nn
import torch.nn.functional as F
from models.detector import HelmetDetector, ANCHORS



def box_iou(b1, b2):

    ix1 = torch.max(b1[..., 0], b2[..., 0])
    iy1 = torch.max(b1[..., 1], b2[..., 1])
    ix2 = torch.min(b1[..., 2], b2[..., 2])
    iy2 = torch.min(b1[..., 3], b2[..., 3])

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a1    = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    a2    = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = a1 + a2 - inter + 1e-7
    return inter / union


def box_giou(pred, target):
    """
    GIoU Loss = 1 - GIoU
    pred, target: (N, 4) x1y1x2y2
    """
    iou = box_iou(pred, target)

    ex1 = torch.min(pred[:, 0], target[:, 0])
    ey1 = torch.min(pred[:, 1], target[:, 1])
    ex2 = torch.max(pred[:, 2], target[:, 2])
    ey2 = torch.max(pred[:, 3], target[:, 3])

    enc_area = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + 1e-7

    a1 = (pred[:,2]-pred[:,0])   * (pred[:,3]-pred[:,1])
    a2 = (target[:,2]-target[:,0]) * (target[:,3]-target[:,1])
    union = a1 + a2 - iou * (a1 + a2 - a1)   # recalc from iou

    giou = iou - (enc_area - (a1 + a2 - iou*(a1+a2-a1))) / enc_area
    return 1 - giou   # loss = 1 - GIoU



class FocalLoss(nn.Module):
 

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



class TargetBuilder:
  
    
    def __init__(self, anchors_list, num_classes=4, img_size=416):
        self.anchors_list = anchors_list   # 3 scales ke anchors
        self.num_classes  = num_classes
        self.img_size     = img_size
        self.registered_anchors = []
        for anchors in anchors_list:
            self.registered_anchors.append(torch.tensor(anchors, dtype=torch.float32))

    def build(self, gt_boxes_list, gt_labels_list, device):
        """
        gt_boxes_list : list of (N_i, 4) tensors — normalized cx,cy,w,h
        gt_labels_list: list of (N_i,)   tensors — class ids
        
        Returns list of (obj_target, cls_target, box_target, pos_mask) per scale
        """
        scales     = [52, 26, 13]
        batch_size = len(gt_boxes_list)
        targets    = []

        for scale_idx, (anchors, grid_size) in enumerate(
                zip(self.anchors_list, scales)):
            
            A  = len(anchors)
            C  = self.num_classes
            S  = grid_size

            obj_t = torch.zeros(batch_size, A, S, S,     device=device)
            cls_t = torch.zeros(batch_size, A, S, S, C,  device=device)
            box_t = torch.zeros(batch_size, A, S, S, 4,  device=device)
            mask  = torch.zeros(batch_size, A, S, S,     device=device,
                                dtype=torch.bool)

            stride = self.img_size / S
            anc_t  = self.registered_anchors[scale_idx].to(device)

            for b in range(batch_size):
                gt_boxes  = gt_boxes_list[b].to(device)   # (N,4) cx cy w h (0-1)
                gt_labels = gt_labels_list[b].to(device)

                if len(gt_boxes) == 0:
                    continue

                px_boxes = gt_boxes.clone()
                px_boxes[:, [0, 2]] *= self.img_size
                px_boxes[:, [1, 3]] *= self.img_size

                for n in range(len(px_boxes)):
                    cx, cy, bw, bh = px_boxes[n]
                    label          = gt_labels[n].item()

                    gi = int(cx / stride)
                    gj = int(cy / stride)
                    gi = min(gi, S - 1)
                    gj = min(gj, S - 1)

                    anchor_iou = torch.min(
                        torch.stack([bw, bh], dim=0).unsqueeze(0).expand(A, -1),
                        anc_t
                    ).prod(dim=1) / (
                        bw * bh + anc_t.prod(dim=1)
                        - torch.min(
                            torch.stack([bw, bh]).unsqueeze(0).expand(A,-1),
                            anc_t
                          ).prod(dim=1) + 1e-7
                    )
                    best_a = anchor_iou.argmax().item()

                    obj_t[b, best_a, gj, gi]      = 1.0
                    cls_t[b, best_a, gj, gi, label] = 1.0
                    mask[b,  best_a, gj, gi]      = True

                    tx = cx / stride - gi
                    ty = cy / stride - gj
                    tw = torch.log(bw / anc_t[best_a, 0] + 1e-7)
                    th = torch.log(bh / anc_t[best_a, 1] + 1e-7)
                    box_t[b, best_a, gj, gi] = torch.stack([tx, ty, tw, th])

            targets.append((obj_t, cls_t, box_t, mask))

        return targets



class DetectionLoss(nn.Module):
    """
    Total Loss = λ_obj * obj_loss
               + λ_cls * cls_loss      (positive anchors only)
               + λ_box * giou_loss     (positive anchors only)
    """

    def __init__(self, anchors_list=ANCHORS, num_classes=4,
                 lambda_obj=1.0, lambda_cls=1.0, lambda_box=5.0, model=None):
        super().__init__()
        self.focal          = FocalLoss(alpha=0.25, gamma=2.0)
        self.target_builder = TargetBuilder(anchors_list, num_classes)
        self.lambda_obj     = lambda_obj
        self.lambda_cls     = lambda_cls
        self.lambda_box     = lambda_box
        self.anchors_list   = anchors_list
        self.model          = model  # Store model instance for decode

    def forward(self, outputs, gt_boxes_list, gt_labels_list):
        """
        outputs       : tuple of 3 raw tensors from model
        gt_boxes_list : list[Tensor(N,4)] normalized cx cy w 
        gt_labels_list: list[Tensor(N,)]
        """
        device  = outputs[0].device
        targets = self.target_builder.build(gt_boxes_list, gt_labels_list, device)

        total_obj = torch.tensor(0., device=device)
        total_cls = torch.tensor(0., device=device)
        total_box = torch.tensor(0., device=device)

        for i, (raw_out, anchors) in enumerate(zip(outputs, self.anchors_list)):
            obj_t, cls_t, box_t, pos_mask = targets[i]

            if self.model is not None:
                decoded_boxes, pred_obj, pred_cls = self.model.decode(
                    raw_out, anchors
                )
            else:
                decoded_boxes, pred_obj, pred_cls = HelmetDetector.decode(
                    HelmetDetector(), raw_out, anchors
                )

            total_obj = total_obj + self.focal(
                pred_obj,          # (B,A,H,W)
                obj_t
            )

            n_pos = pos_mask.sum().item()
            if n_pos > 0:
                pred_cls_pos = pred_cls[pos_mask]   # (N_pos, C)
                cls_t_pos    = cls_t[pos_mask]       # (N_pos, C)
                total_cls    = total_cls + self.focal(pred_cls_pos, cls_t_pos)

                pred_box_pos = decoded_boxes[pos_mask]  # (N_pos, 4) x1y1x2y2
                B, A, H, W = pos_mask.shape
                stride = 416 / H
                gi_idx = pos_mask.nonzero(as_tuple=False)

                anc_t = self.target_builder.registered_anchors[i].to(device)
                b_idx, a_idx, j_idx, i_idx = (
                    pos_mask.nonzero(as_tuple=True)
                )
                tx = box_t[..., 0][pos_mask]
                ty = box_t[..., 1][pos_mask]
                tw = box_t[..., 2][pos_mask]
                th = box_t[..., 3][pos_mask]

                aw = anc_t[a_idx, 0]
                ah = anc_t[a_idx, 1]

                bx = (tx + i_idx.float()) * stride
                by = (ty + j_idx.float()) * stride
                bw = torch.exp(tw) * aw
                bh = torch.exp(th) * ah

                gt_x1 = bx - bw / 2
                gt_y1 = by - bh / 2
                gt_x2 = bx + bw / 2
                gt_y2 = by + bh / 2
                gt_boxes_px = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)

                giou_loss = box_giou(pred_box_pos, gt_boxes_px)
                total_box = total_box + giou_loss.mean()

        total_loss = (self.lambda_obj * total_obj +
                      self.lambda_cls * total_cls +
                      self.lambda_box * total_box)

        return total_loss, total_obj.item(), total_cls.item(), total_box.item()
