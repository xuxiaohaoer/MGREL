import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    """文本分类，RNN模型"""

    def __init__(self, input_size, hidden_num):

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_num
        # 三个待输入的数据

        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size= self.hidden_size, num_layers=1, batch_first=True)
        # self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        # self.f1 = nn.Sequential(nn.Linear(256, 128),
        #                         nn.Dropout(0.8),
        #                         nn.ReLU())
        # self.f1 = nn.Sequential(nn.Linear(128, 2),
        #                         nn.Softmax())

        self.drop = nn.Dropout(p=0.3)
        self.f1 = nn.Linear(self.hidden_size, 84)
        self.f2 = nn.Linear(84, 2)

        # self.f2 = nn.Sigmoid()

    def forward(self, x):
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((x[0], hidden[0]), 1)), dim=1)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                          encoder_outputs.unsqueeze(0))



        output, (hn, cn)= self.rnn(x)

  
        output = output[:,-1,:]
        
        tem = self.drop(output)
        tem = F.relu(self.f1(tem))
        tem = self.f2(tem)
        res = tem
        # res = F.softmax(predict, dim=1)


        # result = predict.squeeze(0)
        return res