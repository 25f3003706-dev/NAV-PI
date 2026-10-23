import torch
import torch.nn as nn

class DetectionLSTM(nn.Module):
    """
    Takes a sequence of per-frame detection vectors and maintains short-term memory.
    Input shape: (B, T, feature_dim)
    feature_dim example: [cls_id, conf, cx, cy, w, h] (+ optional embedding)
    """
    def __init__(self, feature_dim=16, hidden_size=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, feature_dim)  # reconstruct / predict next

    def forward(self, x, hidden=None):
        # x: (B, T, F)
        out, hidden = self.lstm(x, hidden)
        pred = self.fc(out[:, -1, :])  # last timestep
        return pred, hidden, out