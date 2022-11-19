import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.multihead_attention import MultiHeadAttention

class mult_bilstm(nn.Module):
    def __init__(self, word_num, word_len, hidden_num, num_heads, num_layers):
        super(mult_bilstm, self).__init__()
        self.input_size = word_len
        self.hidden_size = hidden_num
        self.num_heads = num_heads

        self.multAtt = MultiHeadAttention(word_len, self.num_heads)
        self.rnn = nn.LSTM(input_size=word_len, hidden_size = self.hidden_size, num_layers = num_layers, batch_first = True, bidirectional = True, dropout = 0.3)
        self.f1 = nn.Linear(self.hidden_size *2, 84)
        self.f2 = nn.Linear(84, 2)
        self.drop = nn.Dropout(0.3)
        
        

    def forward(self, input):
        # input = input[:,:,2:]
        context, att = self.multAtt(input, input, input)
        output, (hn, cn) = self.rnn(context)
        # output = hn.permute(1, 0, 2)
        output = output[:,-1,:]
        tem = self.drop(output)
        tem = F.relu(self.f1(tem))
        tem = self.f2(tem)
        res = tem
        # res = F.softmax(tem, dim =1)
        return res