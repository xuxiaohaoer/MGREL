import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.multihead_attention import MultiHeadAttention
class mult_att_lstm(nn.Module):
    def __init__(self , input_size, feature_num, hidden_num, nums_head):
        super(mult_att_lstm, self).__init__()
        self.input_size = input_size
        self.feature_num = feature_num
        self.hidden_num = hidden_num
        self.nums_head = nums_head

        self.multAtt = MultiHeadAttention(feature_num, self.nums_head)
        self.rnn = nn.LSTM(input_size= self.input_size, hidden_size=hidden_num, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.3)
        self.f1 = nn.Linear(hidden_num, 84)
        self.f2 = nn.Linear(84, 2)

    def forward(self, input):
        context, att = self.multAtt(input, input, input)
        context = context.permute(0, 2, 1)
        output, (final_hidden_state, final_cell_state) = self.rnn(context)
        tem1 = self.f1(final_hidden_state.squeeze(0))
        res =  F.softmax(self.f2(tem1), dim=1)
        return res



