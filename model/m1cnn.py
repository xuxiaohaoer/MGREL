import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class m1cnn(nn.Module):
    def __init__(self):
        super(m1cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=25, stride=1, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(3, stride = 3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=25, stride=1, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(3,stride = 3, padding=1)
        )
        self.f1 = nn.Linear(88* 64, 1024)
        self.f2 = nn.Linear(1024, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, 784)
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = conv2.view(x.size(0), 88*64, -1).squeeze(2)
        output = self.f2(self.f1(conv2))
        return output