import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU — basic unit"""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)   # LeakyReLU detection mein better hai
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
 
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.block = nn.Sequential(
            ConvBNReLU(channels, mid,      kernel=1, padding=0),
            ConvBNReLU(mid,      channels, kernel=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)   # Skip connection


def make_stage(in_ch, out_ch, num_blocks):
    """Ek stage = downsample conv + N residual blocks"""
    layers = [ConvBNReLU(in_ch, out_ch, stride=2)]   # Spatial size halve
    for _ in range(num_blocks):
        layers.append(ResBlock(out_ch))
    return nn.Sequential(*layers)


class CustomBackbone(nn.Module):
    """
    DarkNet-inspired architecture 

    Input  : (B, 3,    416, 416)
    -------
    Stage1 : (B, 64,   208, 208)
    Stage2 : (B, 128,  104, 104)
    Stage3 : (B, 256,   52,  52)  ← P3 (large objects)
    Stage4 : (B, 512,   26,  26)  ← P4 (medium objects)
    Stage5 : (B, 1024,  13,  13)  ← P5 (small objects)
    """

    def __init__(self):
        super().__init__()
        self.stem   = ConvBNReLU(3, 32, kernel=3, stride=1)

        self.stage1 = make_stage(32,   64,   1)
        self.stage2 = make_stage(64,   128,  2)
        self.stage3 = make_stage(128,  256,  4)   # → P3
        self.stage4 = make_stage(256,  512,  4)   # → P4
        self.stage5 = make_stage(512,  1024, 2)   # → P5

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        x  = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5
