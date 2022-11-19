from model.multihead_attention import MultiHeadAttention
from unicodedata import bidirectional
import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F

class HST_MHSA(nn.Module):
    def __init__(self, input_size, feature_num, hidden_num):
        super(HST_MHSA, self).__init__()
        self.input_size = input_size
        self.feature_num = feature_num
        self.hidden_num = hidden_num
        self.embedding = nn.Embedding(257, 128)
        self.rnn_1 = nn.LSTM(input_size=128, hidden_size = 64, num_layers =1, batch_first = True, bidirectional = True)
        self.text_cnn = TextCNN()
        filter_sizes = [3,4,5]
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 128, (size, 128)) for size in filter_sizes])
        self.max_all = nn.AdaptiveMaxPool2d((128, 1))
        self.rnn_2 = nn.LSTM(input_size=128, hidden_size=64, num_layers =1, batch_first=True, bidirectional= True)
        self.multAtt = MultiHeadAttention(128, 2)
        self.mean_all = nn.AdaptiveAvgPool2d((2, 1))


    def forward(self, input):
        batch_size = input.shape[0]
        packet_num = input.shape[1]
        embedding = self.embedding(input).reshape(-1, 100, 128)
        embedding_lstm, (hn, cn)= self.rnn_1(embedding)
        embedding = embedding.reshape(batch_size, packet_num, 100, 128)
        embedding_lstm = embedding_lstm.reshape(batch_size, packet_num, 100, 128)
       
        embedding_cat = torch.cat((embedding_lstm, embedding), dim=2).reshape(-1, 200, 128).unsqueeze(1)
        packet_conv = [F.relu(conv(embedding_cat)).squeeze(3) for conv in self.convs]
        packet_text = torch.cat([F.max_pool1d(item, item.size(2)) for item in packet_conv], 1)
        text = self.max_all(packet_text).squeeze(2).reshape(batch_size, packet_num, 128)

        
        text_rnn, (hn, cn) = self.rnn_2(text)
        context, att = self.multAtt(text_rnn, text_rnn, text_rnn)
        output = self.mean_all(context)

        return  output.squeeze(2)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        input_size = 100
        out_size = 128
        feature_num = 128
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(feature_num - 3 + 1  )
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(feature_num - 4 + 1 )
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(feature_num -5 + 1)
        )
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, input):
        conv_3 = self.conv_block_3(input)
        conv_4 = self.conv_block_4(input)
        conv_5 = self.conv_block_5(input)
        conv = torch.cat((conv_3, conv_4, conv_5), 1)
        output = self.maxpool(conv.permute(0, 2, 1))

        return output
