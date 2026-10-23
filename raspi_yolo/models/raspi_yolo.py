import torch
import torch.nn as nn
from .backbone import MobileNetV3SmallBackbone #change this in your design
from .neck import SimplePAN
from .detect_head import DetectHead
from .lstm_memory import DetectionLSTM

class RaspiYOLO(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.25, lstm_seq=12, feat_dim=16):
        super().__init__()
        self.backbone = MobileNetV3SmallBackbone(width_mult)
        self.neck = SimplePAN(self.backbone.out_channels)
        self.head = DetectHead(self.backbone.out_channels, num_classes)
        self.lstm = DetectionLSTM(feature_dim=feat_dim, hidden_size=128)
        self.num_classes = num_classes
        self.seq_len = lstm_seq
        self.feat_dim = feat_dim

        # Simple buffer for inference-time memory (not part of graph)
        self.register_buffer('frame_buffer', torch.zeros(1, lstm_seq, feat_dim))
        self.hidden = None

    def forward(self, x, mode='detect'):
        """
        mode = 'detect'  → normal YOLO outputs
        mode = 'temporal'→ also runs LSTM on current detection summary
        """
        feats = self.backbone(x)
        feats = self.neck(feats)
        det_outs = self.head(feats)

        if mode == 'detect':
            return det_outs

        # Build a compact detection vector for the current frame (simplified)
        # In practice you decode boxes first, then pack top-k into fixed vector
        # Here we use a global average as placeholder for demo
        summary = torch.zeros(x.size(0), self.feat_dim, device=x.device)
        # ... (real code would decode and pack detections)

        # Slide buffer
        self.frame_buffer = torch.cat([self.frame_buffer[:, 1:], summary.unsqueeze(1)], dim=1)
        pred, self.hidden, full = self.lstm(self.frame_buffer, self.hidden)
        return det_outs, pred, full

    def reset_memory(self):
        self.frame_buffer.zero_()
        self.hidden = None