import torch
import torch.nn as nn

class DetectHead(nn.Module):
    def __init__(self, in_channels, num_classes, reg_max=1):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.no = num_classes + 4  # cls + box (simplified)

        self.heads = nn.ModuleList()
        for c in in_channels:
            self.heads.append(nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, self.no, 1)
            ))

    def forward(self, features):
        outputs = []
        for i, x in enumerate(features):
            outputs.append(self.heads[i](x))
        return outputs