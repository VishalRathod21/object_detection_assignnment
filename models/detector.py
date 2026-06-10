import torch
import torch.nn as nn
from models.backbone import ConvBNReLU, CustomBackbone

ANCHORS = [
    [(10, 13),  (16, 30),   (33, 23)],    # Small objects  → 52x52 head
    [(30, 61),  (62, 45),   (59, 119)],   # Medium objects → 26x26 head
    [(116, 90), (156, 198), (373, 326)],  # Large objects  → 13x13 head
]


class DetectionHead(nn.Module):
    """
    Single scale ke liye detection head.
    Output per anchor: tx, ty, tw, th, objectness, c0, c1, c2, c3
                                                    (4 classes)
    """

    def __init__(self, in_ch, num_anchors=3, num_classes=4):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)

        self.neck = nn.Sequential(
            ConvBNReLU(in_ch,      in_ch // 2, kernel=1, padding=0),
            ConvBNReLU(in_ch // 2, in_ch,      kernel=3, padding=1),
            ConvBNReLU(in_ch,      in_ch // 2, kernel=1, padding=0),
            ConvBNReLU(in_ch // 2, in_ch,      kernel=3, padding=1),
        )
        self.pred = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pred(self.neck(x))


class HelmetDetector(nn.Module):
    """
    Full detection model:
      CustomBackbone → 3 DetectionHeads (multi-scale)
    """

    IMG_SIZE = 416
    ANCHORS = ANCHORS

    def __init__(self, num_classes=4):
        super().__init__()

        self.NUM_CLASSES = num_classes

        self.backbone = CustomBackbone()

        self.head_large = DetectionHead(
            256,
            num_classes=self.NUM_CLASSES
        )

        self.head_medium = DetectionHead(
            512,
            num_classes=self.NUM_CLASSES
        )

        self.head_small = DetectionHead(
            1024,
            num_classes=self.NUM_CLASSES
        )
        
        self.register_buffer('anchors_large', torch.tensor(self.ANCHORS[0], dtype=torch.float32))
        self.register_buffer('anchors_medium', torch.tensor(self.ANCHORS[1], dtype=torch.float32))
        self.register_buffer('anchors_small', torch.tensor(self.ANCHORS[2], dtype=torch.float32))

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)

        out_l = self.head_large(p3)
        out_m = self.head_medium(p4)
        out_s = self.head_small(p5)

        return out_l, out_m, out_s

    def decode(self, output, anchors, img_size=416):
        """
        Raw output → decoded predictions
        """

        device = output.device

        B, _, H, W = output.shape
        A = len(anchors)
        C = 4

        out = (
            output.view(B, A, 5 + C, H, W)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        stride = img_size / H

        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )

        gx = gx.view(1, 1, H, W)
        gy = gy.view(1, 1, H, W)

        if not isinstance(anchors, torch.Tensor):
            anc = torch.tensor(anchors, device=device, dtype=torch.float32)
        else:
            anc = anchors.to(device).float()

        aw = anc[:, 0].view(1, A, 1, 1)
        ah = anc[:, 1].view(1, A, 1, 1)

        bx = (torch.sigmoid(out[..., 0]) + gx) * stride
        by = (torch.sigmoid(out[..., 1]) + gy) * stride

        bw = torch.exp(out[..., 2]) * aw
        bh = torch.exp(out[..., 3]) * ah

        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2

        obj = torch.sigmoid(out[..., 4])
        cls = torch.sigmoid(out[..., 5:])

        decoded = torch.stack(
            [x1, y1, x2, y2],
            dim=-1
        )

        return decoded, obj, cls