import torch.nn as nn
import torch 
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.modules import dropout

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.input_size = 300
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size = self.hidden_size, num_layers =1, batch_first = True, bidirectional = True, dropout = 0.3)
        self.f1 = nn.Linear(self.hidden_size *2, 84)
        self.f2 = nn.Linear(84, 2)
        self.drop = nn.Dropout(0.3)
    
    def forward(self, x):
        # x = x[:,-30:,:5]
        output, (hn, cn) = self.rnn(x)
        # output = hn.permute(1, 0, 2)
        output = output[:,-1,:]
        
        tem = self.drop(output)
        tem = F.relu(self.f1(tem))
        tem = self.f2(tem)
        res = tem
        # res = F.softmax(tem, dim =1)
        return res

