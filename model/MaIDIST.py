from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaIDIST(nn.Module):
    def __init__(self):
        super(MaIDIST, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv1d(1, 16, 25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Conv1d(16, 32, 25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=1),
            nn.Dropout(0.2),
        )
        self.gru = nn.GRU(input_size = 3, hidden_size = 64, batch_first=True, bidirectional=True)
        self.f1 = nn.Linear(32*88, 128)
        self.m2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size = 14, hidden_size=65, batch_first = True, bidirectional=True)
        self.m3 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size = (3, 3), stride =1, padding=(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride =1, padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1, 1),
        )
        self.m4 = nn.Sequential(
            nn.Linear(64 *5* 130, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.m5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0,2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.f2 = nn.Linear(64, 128)
    def forward(self, x):
        mix_1 = x[:,0,:28,:28].view(x.shape[0], -1, 28*28)
        mix_2 = x[:,1,:,:3]
        mix_3 = x[:,2,:5,:14]
        mix_1 = self.m1(mix_1)
        mix_1 = F.relu(self.f1(mix_1.view(x.shape[0], 32*88, -1).squeeze(2)))
        output_1, (hn, cn) = self.gru(mix_2)
        mix_2 = self.m2(hn)
        output_2, (hn_1, cn_1) = self.lstm(mix_3)
        output_2 = output_2.unsqueeze(1)
        tem = self.m3(output_2).view(x.shape[0], 64*5*130, -1, 1).squeeze(2).squeeze(2)
        mix_3 = self.m4(tem)
        x = torch.cat((mix_1, mix_2, mix_3), dim=1)
        # x = torch.cat((mix_1, mix_2), dim=1)
        output = self.m5(x)
        return output
