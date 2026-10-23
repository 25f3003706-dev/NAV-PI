import torch
import torch.nn as nn
from .backbone import ConvBNAct

class SimplePAN(nn.Module):
    """Lightweight Path Aggregation Network."""
    def __init__(self, in_channels):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lateral5 = ConvBNAct(c5, c4, 1)
        self.lateral4 = ConvBNAct(c4, c3, 1)
        self.down4 = ConvBNAct(c3, c3, 3, 2)
        self.down5 = ConvBNAct(c4, c4, 3, 2)
        self.out3 = ConvBNAct(c3, c3, 3)
        self.out4 = ConvBNAct(c3 + c4, c4, 3)
        self.out5 = ConvBNAct(c4 + c5, c5, 3)

    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + nn.functional.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = c3 + nn.functional.interpolate(p4, size=c3.shape[2:], mode='nearest')

        n3 = self.out3(p3)
        n4 = self.out4(torch.cat([self.down4(n3), p4], 1))
        n5 = self.out5(torch.cat([self.down5(n4), p5], 1))
        return n3, n4, n5