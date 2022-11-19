import torch
import torch.nn as nn
import torch.nn.functional as F
from model.multihead_attention import MultiHeadAttention

class RTETC(nn.Module):
    def __init__(self, input_size, feature_num, hidden_num):
        super(RTETC, self).__init__()
        self.input_size = input_size
        self.feature_num = feature_num
        self.hidden_num = hidden_num

        # self.embed_1 = nn.Embedding(11, 144)
        # self.embed_2 = nn.Embedding(1515, 144) //datacon
        self.embed_1 = nn.Embedding(15, 144)
        self.embed_2 = nn.Embedding(5846, 144)
        self.linear = nn.Linear(144, 144)
        self.multAtt_1 = MultiHeadAttention(432, 3)
        self.fres_1 = nn.Linear(432, 432)
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.multAtt_2 = MultiHeadAttention(432, 3)
        self.fres_2 = nn.Linear(432, 432)
        self.conv_2 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.f1 = nn.Linear(1296, 1024)
        self.f2 = nn.Linear(1024, 128)
        self.f3 = nn.Linear(128, 2)
    
    def forward(self, input):
        embedding_1 = self.embed_1(input[:,:,0])
        embedding_2 = self.embed_2(input[:,:,1])
        embedding_3 = self.linear(input[:,:,2:].float())
        embedding = torch.cat((embedding_1, embedding_2, embedding_3), dim=2)
        context_1, att_1 = self.multAtt_1(embedding, embedding, embedding)
        context_res = context_1 + self.fres_1(embedding)
        conv_1 = self.conv_1(context_res)
        context_2, att_2 = self.multAtt_2(conv_1, conv_1, conv_1)
        context_res_2 = context_2 + self.fres_2(conv_1)
        conv_2 = self.conv_2(context_res_2).reshape(input.shape[0], -1)
        output = self.f3(F.relu(self.f2(F.relu(self.f1(conv_2)))))

        return output
