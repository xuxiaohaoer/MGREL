import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.multihead_attention import MultiHeadAttention

class mix_mult_bilstm_2(nn.Module):
    def __init__(self, word_len_1, word_len_2, hidden_num, num_heads, num_layers, args):
        super(mix_mult_bilstm_2, self).__init__()

        self.hidden_size = hidden_num
        self.num_heads = num_heads
        self.word_len_1 = word_len_1
        self.word_len_2 = word_len_2 
        self.multAtt = MultiHeadAttention(6, num_heads)
        self.rnn1 = nn.LSTM(input_size=6, hidden_size = self.hidden_size, num_layers = args.num_layers, batch_first = True, bidirectional = True, dropout = 0.3)
        if args.f == "mix_word_seq_pay_ip":
            self.rnn2 = nn.LSTM(input_size=100, hidden_size = self.hidden_size, num_layers = args.num_layers_2, batch_first = True, bidirectional = True, dropout = 0.3)
            self.feature = args.f
        elif args.f == "mix_word_seq_ip" or args.f == "mix_word_seq_1_ip" or args.f == "mix_word_seq":
            self.rnn2 = nn.LSTM(input_size=5, hidden_size = self.hidden_size, num_layers = args.num_layers_2, batch_first = True, bidirectional = True, dropout = 0.3)
            self.feature = args.f
        
        
        self.a = nn.Parameter(torch.rand(1))
        self.f1 = nn.Linear(self.hidden_size * 4, 256)
        self.f2 = nn.Linear(256, 84)
        self.f3 = nn.Linear(84, 2)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, input):
        if self.feature == "mix_word_seq_ip" or self.feature == "mix_word_seq_1_ip" or self.feature == "mix_word_seq":
            word_seq = input[:,:self.word_len_1]
            mult_seq = input[:,-30:,:5]
        else:
            word_seq = input[:,:self.word_len_1,:6]
            mult_seq = input[:,-10:,]
        # word_seq = input[:,:-10,:6]
        # mult_seq = input[:,-10:,]

        context, att = self.multAtt(word_seq, word_seq, word_seq)
        output_word, (hn, cn) = self.rnn1(context)
        # output = hn.permute(1, 0, 2)
        output_word = output_word[:,-1,:]
        word = self.drop1(output_word)

        output_mult, (hn, cn) = self.rnn2(mult_seq)
        # output = hn.permute(1, 0, 2)
        output_mult = output_mult[:,-1,:]
        mult = self.drop2(output_mult)

        output = torch.cat((word, mult), -1)

        tem = F.relu(self.f1(output))
        # tem = F.relu(self.f3(F.relu(self.f2(tem))))
        tem = self.f3(F.relu(self.f2(tem)))
        res = tem
        return res
        



        