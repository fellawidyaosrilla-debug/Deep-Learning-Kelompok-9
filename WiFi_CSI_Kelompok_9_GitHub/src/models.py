"""Arsitektur MLP dan CNN-LSTM identik dengan notebook Kelompok 9."""
import torch
from torch import nn
from .config import WINDOW, CHANNELS, SUBCARRIERS


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(CHANNELS * WINDOW * SUBCARRIERS, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((WINDOW, 5))
        )
        self.lstm = nn.LSTM(input_size=32 * 5, hidden_size=64, batch_first=True)
        self.regressor = nn.Linear(64, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1, 3).contiguous().flatten(2)
        _, (h, _) = self.lstm(x)
        return self.regressor(h[-1])


def make_model(name):
    return {'MLP': MLP, 'CNN-LSTM': CNNLSTM}[name]()
