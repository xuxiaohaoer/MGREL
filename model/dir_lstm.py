import torch
import torch.nn as nn
import torch.nn.functional as F

class dir_lstm(nn.Module):
    def __init__(self, word_num=15, word_len=4, hidden_num=144):
        super(dir_lstm, self).__init__()
        self.rnn1 = nn.LSTM(input_size = word_len, hidden_size=hidden_num, batch_first = True)
        self.rnn2 = nn.LSTM(input_size = word_len, hidden_size=hidden_num, batch_first = True)
        self.drop = nn.Dropout(0.3)
        self.f1 = nn.Linear(hidden_num *2, 84)
        self.f2 = nn.Linear(84, 2)
    

    def forward(self, input):
        src = input[:,0]
        dst = input[:,1]

        output_1, (hn_1, cn_1) = self.rnn1(src)
        output_2, (hn_2, cn_2) = self.rnn2(dst)

        embedding = torch.cat((hn_1.squeeze(0), hn_2.squeeze(0)), dim=1)
        output = self.f2(F.relu(self.f1(self.drop(embedding))))


        return output 