import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand=6):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, hidden, 1))
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV3SmallBackbone(nn.Module):
    """Ultra-light backbone suitable for RPi."""
    def __init__(self, width_mult=0.25):
        super().__init__()
        def c(ch): return max(8, int(ch * width_mult))

        self.stem = ConvBNAct(3, c(16), 3, 2)
        self.stage1 = InvertedResidual(c(16), c(16), 2, expand=1)
        self.stage2 = nn.Sequential(
            InvertedResidual(c(16), c(24), 2),
            InvertedResidual(c(24), c(24), 1),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(c(24), c(40), 2),
            InvertedResidual(c(40), c(40), 1),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(c(40), c(80), 2),
            InvertedResidual(c(80), c(80), 1),
            InvertedResidual(c(80), c(80), 1),
        )
        self.out_channels = [c(24), c(40), c(80)]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)      # stride 8
        c4 = self.stage3(c3)     # stride 16
        c5 = self.stage4(c4)     # stride 32
        return c3, c4, c5