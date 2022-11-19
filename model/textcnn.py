import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import ReLU
class text_cnn(nn.Module):
    def __init__(self, input_size, feature_num, out_size):
        super(text_cnn, self).__init__()
        self.input_size = input_size
        self.feature_num = feature_num



        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(feature_num - 2 + 1)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(feature_num - 3 +1)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(feature_num -4 +1)
        )
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(out_size * 3, 2)
        # self.multAtt2 = MultiHeadAttention(feature_num, self.nums_head)
        # self.f2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        # self.drop3 = nn.Dropout(0.3)
        # self.f3 = nn.Linear(144, 2)
    def forward(self, input):


        # tem = F.relu(self.f0(input))
        # tem = self.drop1(tem)

        
        conv_2 = self.conv_block_2(input).squeeze(2)
        conv_3 =self.conv_block_3(input).squeeze(2)
        conv_4 = self.conv_block_4(input).squeeze(2)

        conv = torch.cat((conv_2, conv_3, conv_4), 1)

        res = self.fc(conv)
        res = F.softmax(res, dim=1)
        # cov_1 = F.relu(self.f1(context))
        # cov_1 = self.drop2(cov_1)
        # res = self.f3(cov_1)
        # res = res.squeeze(1)
        # res = F.softmax(res, dim=1)
        
        return res
