import sys

from numpy.lib.index_tricks import diag_indices_from
from model.mult_att_lstm import mult_att_lstm
from pre_data import pre_dataset
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import copy
import numpy as np
import torch.nn as nn
from torch import optim
from model_att import LSTM_Attention
from sklearn.metrics import roc_auc_score

# from nnModel import mult_att_CNN
from model.mult_att_1d_CNN import  mult_att_CNN
from model.mult_att_lstm_att import mult_att_lstm_att
from model.bilstm import BiLSTM
from model.textcnn import text_cnn
from model.mult_bilstm import mult_bilstm
from model.cnn import CNN
from model.lstm import LSTM
from model.dir_lstm import dir_lstm
import time
from model.HST_MHSA import HST_MHSA
from model.mix_mult_bilstm import mix_mult_bilstm
from model.mix_mult_bilstm_2 import mix_mult_bilstm_2
from model.RTETC import RTETC
from model.m1cnn import m1cnn
from model.m2cnn import m2cnn
from model.deepMal import deepMal
from model.MaIDIST import MaIDIST
# from ignite.handlers import  EarlyStopping

from index import cal_index
from tqdm import tqdm
import os
import random
import argparse
from focal_loss import FocalLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
from focal_loss import FocalLoss
print("device:{}".format(device))
# seed = 66
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
class DS():
  
    def __init__(self, args):
        
        self.batch_size = args.b
        self.model_use = args.m
        self.learning_rate = args.l
        self.hidden_num = args.h

        fixed_random(66) # 固定训练集和测试集的随机参数

        self.dataloaders_dict, self.dataset_sizes = pre_dataset(args)
        word_num =  self.dataloaders_dict["train"].dataset.tensors[0].shape[1]
        word_len = self.dataloaders_dict["test"].dataset.tensors[0].shape[2]
        model_use = args.m
        print("model:", model_use)
        print("word_num:", word_num, "word_len:", word_len)
        fixed_random(66) # 固定训练过程
        if model_use == 'lstm_att':
            self.model = LSTM_Attention(word_num, word_len, hidden_dim=self.hidden_num, n_layers=1).to(device)
        elif model_use == 'mult_att_cnn':
            self.model = mult_att_CNN(word_num, word_len, out_size=5,kernel_size=3, nums_head=2).to(device)
        elif model_use == "mult_att_lstm":
            self.model = mult_att_lstm(word_num, word_len, self.hidden_num, nums_head=2).to(device)
        elif model_use =='mult_att_lstm_att':
            self.model = mult_att_lstm_att(word_num, word_len, self.hidden_num, nums_head=2).to(device)
        elif model_use =='lstm':
            self.model = LSTM(word_len, hidden_num=self.hidden_num).to(device)
        elif model_use =='bilstm':
            self.model = BiLSTM(word_len, hidden_size=self.hidden_num).to(device)
        elif model_use =='text_cnn':
            self.model = text_cnn(word_num, word_len, out_size = 128).to(device)
        elif model_use =='mult_bilstm':
            self.model = mult_bilstm(word_num, word_len, hidden_num=self.hidden_num, num_heads=args.num_heads, num_layers=args.num_layers).to(device)
        elif model_use =='cnn':
            self.model = CNN().to(device)
        elif model_use == 'HST_MHSA':
            self.model = HST_MHSA(word_num, word_len, hidden_num=self.hidden_num).to(device)
        elif model_use == 'RTETC':
            self.model = RTETC(word_num, word_len, hidden_num=self.hidden_num).to(device) 
        elif model_use == 'dir_lstm':
            self.model =  dir_lstm(hidden_num = self.hidden_num).to(device) 
        elif model_use == "mix_mult_bilstm":
            self.model = mix_mult_bilstm(word_len_1 = args.word_num, word_len_2 = 30, hidden_num=self.hidden_num,  num_heads=args.num_heads, num_layers=args.num_layers, args=args).to(device)
        elif model_use == "mix_mult_bilstm_2":
            self.model = mix_mult_bilstm_2(word_len_1 = args.word_num, word_len_2 = 30, hidden_num=self.hidden_num,  num_heads=args.num_heads, num_layers=args.num_layers, args=args).to(device)
        elif model_use == "m1cnn":
            self.model = m1cnn().to(device)
        elif model_use == "m2cnn":
            self.model = m2cnn().to(device)
        elif model_use == "deepMal":
            self.model = deepMal().to(device)
        elif model_use == 'MaIDIST':
            self.model = MaIDIST().to(device)


        if args.loss == "CrossEntropyLoss":
            self.Loss = nn.CrossEntropyLoss()
        else:
            self.Loss = FocalLoss(gamma=args.f_g, alpha=args.f_a)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # lr:0.001
        if model_use == "mix_mult_bilstm" or model_use == "mix_mult_bilstm_2":
            model_1 = list(map(id, self.model.multAtt.parameters()))
            model_2 = list(map(id, self.model.rnn1.parameters()))
            base_params = filter(lambda p: id(p) not in model_1 + model_2, self.model.parameters())
            self.optimizer = optim.Adam([{'params': base_params}, 
            {'params': self.model.multAtt.parameters(), 'lr': self.learning_rate}, 
            {'params': self.model.rnn1.parameters(), 'lr': self.learning_rate}], lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # lr:0.001
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.patience = 20
        self.word_num = word_num
        self.word_len = word_len
        self.performation = {}
        self.time_now = time.asctime(time.localtime(time.time()))
        self.dir = "./model_save/{}_{}_{}_{}/{}".format(args.d_1, args.d_2, args.f, args.m, self.time_now)
        print("time:{}".format(self.time_now))
      
       

    def train_model(self, save_path, epoch_num=10):
      
        self.epoch_num = epoch_num
        best_loss = 100
        flag = True
        for epoch in range(epoch_num):
            result = {'train':{'acc':0, 'loss':0},
                    'val':{'acc':0, 'loss':0}}
            # atts = torch.zeros(55, 55).to(device)
            if not flag:
                break
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                runing_loss = 0
                num_correct = 0
                if not flag:
                    break
                for x_batch, y_batch in tqdm(self.dataloaders_dict[phase]):
                    # x_batch = x_batch.unsqueeze(1)
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):

                        out = self.model(x_batch)
                        loss = self.Loss(out, y_batch)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        pred = out.argmax(dim=1)
                    runing_loss += loss.item() * x_batch.size(0)
                    num_correct += torch.eq(pred, y_batch).sum().float().item()

                result[phase]['acc'] = num_correct/self.dataset_sizes[phase]
                result[phase]['loss'] = runing_loss/self.dataset_sizes[phase]

                if phase == 'val':
                    if best_loss <= result[phase]['loss']:
                        inc_num += 1
                    else:
                        inc_num = 0
                    if inc_num == self.patience:
                        print("Early stop")
                        flag = False
                        break
                    
                if phase == 'val' and result[phase]['loss']<best_loss:
                    best_loss = result[phase]['loss']
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
            with open("{}/{}_record.txt".format(save_path, self.time_now), "a+") as f:
                f.write("\n epoch:{}/{} train_acc:{:.3} train_loss:{:.3f} val_acc:{:.3f} val_loss:{:.3}".format(epoch+1, epoch_num, result['train']['acc'], result['train']['loss'], result['val']['acc'], result['val']['loss']))
            print("epoch:{}/{}".format(epoch+1, epoch_num), "train_acc:{:.3}".format(result['train']['acc']), "train_loss:{:.3}".format(result['train']['loss']),
                "val_acc:{:.3}".format(result['val']['acc']), "val_loss:{:.3}".format(result['val']['loss']))
            if epoch + 1 >= 100:
                self.model.eval()
                num_correct = 0
                y_label = []
                y_pred = []
                label_all = []
                prob_all = []
                for step, (x_batch, y_batch)in enumerate(self.dataloaders_dict['test']):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    with torch.no_grad():
                        out = self.model(x_batch)
                        pred = out.argmax(dim=1)
                    num_correct += torch.eq(pred, y_batch).sum().float().item()
                    y_batch = y_batch.cpu().numpy()
                    pred = pred.cpu().numpy()
                    for i in y_batch:
                        y_label.append(i)
                    for i in pred:
                        y_pred.append(i)
                    prob_all.extend(out[:,1].cpu().numpy())
                    label_all.extend(y_batch)
        
                acc, pre, rec, f1, matrix = cal_index(y_label, y_pred)
                auc = roc_auc_score(label_all, prob_all)    
                with open(self.dir + "/" + '{}_ill.txt'.format(self.time_now),'a+',encoding='utf-8') as f:
                    list = []
                    list.append("epoch:{} acc:{:.4f} pre:{:.4f} rec:{:.4f} f1:{:.4f} auc:{:.4f} matrix:{} {}\n".format(epoch+1, acc, pre, rec, f1, auc, matrix[0], matrix[1]))
                    f.writelines(list)
                f.close()
                torch.save(self.best_model_wts, '{}/{}.pth'.format(save_path, epoch+1))
                
        torch.save(self.best_model_wts, '{}/{}.pth'.format(save_path, self.time_now))

    def test_model(self, flag, load_path):
        # model  = nnModel.LSTM().to(device)
        # model = LSTM_Attention(3, 256, 1).to(device)
        
        if flag == "load":
            self.model.load_state_dict(torch.load(load_path))
        else:
            self.model.load_state_dict(self.best_model_wts)
        self.model.eval()
        num_correct = 0
        y_label = []
        y_pred = []
        label_all = []
        prob_all = []
        for step, (x_batch, y_batch)in enumerate(self.dataloaders_dict['test']):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                out = self.model(x_batch)
                pred = out.argmax(dim=1)
            num_correct += torch.eq(pred, y_batch).sum().float().item()
            y_batch = y_batch.cpu().numpy()
            pred = pred.cpu().numpy()
            for i in y_batch:
                y_label.append(i)
            for i in pred:
                y_pred.append(i)
            prob_all.extend(out[:,1].cpu().numpy())
            label_all.extend(y_batch)
        
        acc, pre, rec, f1, matrix = cal_index(y_label, y_pred)
        auc = roc_auc_score(label_all, prob_all)
        print("auc:{:.4}".format(auc))
        self.performation["acc"] = acc
        self.performation["pre"] = pre
        self.performation["rec"] = rec
        self.performation["f1"] = f1
        self.performation["matrix"] = matrix
        self.performation["auc"] = auc



    # print("test:",  num_correct / dataset_sizes['test'])
def dp(argv=["", "datacon_black", "datacon_white", "RTETC", "content_seq", "144"]):
    # model_use = "mult_bilstm"
    # feature_type = "mix"
    """
    dataset_1:第一个数据集
    dataset_2:第二个数据集
    model_use:模型选择
    feature_type:特征种类
    hidden_num:隐藏层数目
    batch_size:
    epoch:
    learning_rate:
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d_1", type=str, default="datacon_black", help="first dataset")
    parser.add_argument("--d_2", type=str, default="datacon_white", help="second dataset")
    parser.add_argument("--m", type=str, default="mix_mult_bilstm", help="select model")
    parser.add_argument("--f", type=str, default="mix_word_seq__ip", help="select feature")
    parser.add_argument("--h", type=int, default=144, help="first hidden_num")
    parser.add_argument("--b", type=int, default=8, help="batch size")
    parser.add_argument("--e", type=int, default=300, help="epoch num")
    parser.add_argument("--l", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_heads", type=int, default=2, help="num_heads")
    parser.add_argument("--num_layers", type=int, default=1, help="num_layers")
    parser.add_argument("--num_layers_2", type=int, default=1, help="num_layser_2")
    parser.add_argument("--a", type=float, default=0.5, help="a")
    parser.add_argument("--loss", type=str, default="CrossEntropyLoss", help="loss")
    parser.add_argument("--f_a", type=float, default=0.0, help="focal_loss alpha")
    parser.add_argument("--f_g", type=float, default=0.0, help="focal_loss gamma")
    parser.add_argument("--word_len", type=int, default=6, help="word len") # test
    parser.add_argument("--word_num", type=int, default=63, help="word num") # test
    args = parser.parse_args()

    system = DS(args)
    
    dir = "./model_save_2/{}_{}_{}_{}/{}".format(args.d_1, args.d_2, args.f, args.m, system.time_now)

    if not os.path.exists(dir): 
        os.makedirs(dir)


    with open(dir + "/" + '{}_ill.txt'.format(system.time_now),'a+',encoding='utf-8') as f:
        list = []
        list.append("dataset_1:{}\n".format(args.d_1))
        list.append("dataset_2:{}\n".format(args.d_2))
        list.append("model_use:{}\n".format(args.m))
        list.append("feature_type:{}\n".format(args.f))
        list.append("hidden_num:{}\n".format(args.h))
        list.append("batch_size:{}\n".format(args.b))
        list.append("learning_rate:{}\n".format(args.l))
        list.append("word_num:{}\n".format(system.word_num))
        list.append("word_len:{}\n".format(system.word_len))
        if args.m == "mix_mult_bilstm":
            list.append("a:{}\n".format(system.model.a))
            list.append("num_heads:{}\n".format(args.num_heads))
            list.append("num_layers:{}\n".format(args.num_layers))
        if args.m == "mult_bilstm":
            list.append("num_heads:{}\n".format(args.num_heads))
            list.append("num_layers:{}\n".format(args.num_layers))
        if args.m == "mix_mult_bilstm_2":
            list.append("num_heads:{}\n".format(args.num_heads))
            list.append("num_layers_1:{}\n".format(args.num_layers))
            list.append("num_layers_2:{}\n".format(args.num_layers_2))
        list.append("loss:{}\n".format(args.loss))
        if args.loss == "focal":
            list.append("a:{}\n".format(args.f_a))
            list.append("gamma:{}\n".format(args.f_g))
        f.writelines(list)
    f.close()
        
    system.train_model(save_path = dir, epoch_num= args.e)
    system.test_model("test", "")
    # system.test_model("load", dir + "/Wed Jul 21 18:10:04 2021.pth")
    with open(dir + "/" + '{}_ill.txt'.format(system.time_now),'a+',encoding='utf-8') as f:
        list = []
        if args.m == "mix_mult_bilstm":
            list.append("a:{}\n".format(system.model.a))
        list.append("acc:{:.4f}\n".format(system.performation["acc"]))
        list.append("pre:{:.4f}\n".format(system.performation["pre"]))
        list.append("rec:{:.4f}\n".format(system.performation["rec"]))
        list.append("f1:{:.4f}\n".format(system.performation["f1"]))
        list.append("auc:{:.4f}\n".format(system.performation["auc"]))
        list.append("matrix:{}\n".format(system.performation["matrix"]))


        f.writelines(list)
    f.close()
    
 




def fixed_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    dp()








