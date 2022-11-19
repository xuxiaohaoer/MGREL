import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class m2cnn(nn.Module):
    def __init__(self):
        super(m2cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (5,5), stride=1, padding=2),
            nn.Softmax(),
            nn.MaxPool2d((2,2))
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 2),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        img = self.conv(x)
        img = img.view(img.shape[0], -1)
        output = self.fc(img)
        return output
