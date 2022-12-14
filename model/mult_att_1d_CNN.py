import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import ReLU
class mult_att_CNN(nn.Module):
    def __init__(self, input_size, feature_num, out_size, kernel_size, nums_head):
        super(mult_att_CNN, self).__init__()
        self.input_size = input_size
        self.feature_num = feature_num
        self.nums_head = nums_head

    
        self.multAtt = MultiHeadAttention(feature_num, self.nums_head)
        
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
        tem = input
        
        context, att = self.multAtt(tem, tem, tem)

        
        conv_2 = self.conv_block_2(context).squeeze(2)
        conv_3 =self.conv_block_3(context).squeeze(2)
        conv_4 = self.conv_block_4(context).squeeze(2)

        conv = torch.cat((conv_2, conv_3, conv_4), 1)

        res = self.fc(conv)
        res = F.softmax(res, dim=1)
        # cov_1 = F.relu(self.f1(context))
        # cov_1 = self.drop2(cov_1)
        # res = self.f3(cov_1)
        # res = res.squeeze(1)
        # res = F.softmax(res, dim=1)
        
        return res



class dot_attention(nn.Module):
    """ ?????????????????????"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        ????????????
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: ??????????????????attention?????????
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # ??????????????????
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # ?????????mask?????????????????????????????????
        # ??????softmax
        attention = self.softmax(attention)
        # ??????dropout
        attention = self.dropout(attention)
        # ???v????????????
        context = torch.bmm(attention, v)
        return context, attention



class MultiHeadAttention(nn.Module):
    """ ??????????????????"""
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads   # ??????????????????
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm ????????????

    def forward(self, key, value, query, attn_mask=None):
        # ????????????
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # ????????????
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # ?????????????????????
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # ???????????????????????????
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # ??????????????? concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # ??????????????????
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # ?????????????????????????????????
        output = self.layer_norm(residual + output)

        return output, attention
