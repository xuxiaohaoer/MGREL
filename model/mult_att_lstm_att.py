import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.multihead_attention import MultiHeadAttention
import math
class mult_att_lstm_att(nn.Module):
    def __init__(self , input_size, feature_num, hidden_num, nums_head):
        super(mult_att_lstm_att, self).__init__()
        self.input_size = input_size
        self.feature_num = feature_num
        self.hidden_num = hidden_num
        self.nums_head = nums_head

        self.multAtt = MultiHeadAttention(feature_num, self.nums_head)
        self.rnn = nn.LSTM(input_size= input_size, hidden_size=hidden_num, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.3)
        self.f1 = nn.Linear(feature_num, 2)



    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)  # d_k为query的维度

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        #         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)

        return context, alpha_n
    def forward(self, input):
        context, att = self.multAtt(input, input, input)
        context = context.permute(0, 2, 1)

        output, (final_hidden_state, final_cell_state) = self.rnn(context)

        output = output.permute(0, 2, 1)
        query = self.drop(output)

        attn_output, alpha_n = self.attention_net(output, query)

     
        res =  F.softmax(self.f1(attn_output))
        return res



