import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class deepMal(nn.Module):
    def __init__(self):
        super(deepMal, self).__init__()
        self.conv = nn.Conv1d(2,32, 5)
        self.fc1 = nn.Linear(32 * 140 ,50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 2)
    

    def forward(self, x):
        conv = self.conv(x)
        conv  = conv.view(conv.shape[0], -1)
        tem = F.relu(self.fc1(conv))
        output  = F.relu(self.fc2(tem))
        return self.fc3(output)