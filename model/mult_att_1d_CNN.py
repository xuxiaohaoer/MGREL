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
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention



class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads   # 每个头的维度
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm 归一化。

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 线性映射
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 按照头进行分割
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 缩放点击注意力机制
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # 进行头合并 concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 进行线性映射
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # 添加残差层和正则化层。
        output = self.layer_norm(residual + output)

        return output, attention
