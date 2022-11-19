import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6,16, kernel_size=5)
        self.conv3 = nn.Conv2d(16,120, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,2)
        self.logsoftmax = nn.LogSoftmax()

    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).float()

        conv1 = F.relu(self.mp(self.conv1(x)))
        conv2 = F.relu(self.mp(self.conv2(conv1)))
        conv3 = F.relu(self.conv3(conv2))
        tem = conv3.view(batch_size, -1)
        out = F.relu(self.fc1(tem))
        
        out = self.logsoftmax(self.fc2(out))
        return out
