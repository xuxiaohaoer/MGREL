import numpy as np 
import torch 
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as Data

# dataset = ["CTU-13", "datacon_black", "datacon_white", "malware-lastline", "malware-mta", "malware-stratosphere", "normal"]

def pre_tlsWord():
    from gensim.models import word2vec
    import torch
    import numpy as np 
    d_black = np.load("data/malware-mta_tlsWord.npy", allow_pickle=True)
    d_white = np.load("data/normal_tlsWord.npy", allow_pickle=True)
    tests = []
    len_count = []
    for dataset in [d_black, d_white]:
        for data in dataset:
            tests.append(data["feature"])
            len_count.append(len(data["feature"]))
    from gensim.models import word2vec
    model = word2vec.Word2Vec(tests, vector_size=300, min_count=1, sg=0) # cbow
    d_black_cbow = []
    d_black_raw = []
    len_base = 32
    for sample in d_black:

        d_black_raw.append(sample['feature'])

    for sample in d_black_raw:
        tem = []
        for value in sample:
            tem.append(list(model.wv.get_vector(value)))
        while len(tem)<len_base:
            tem.append([0] *300)
        tem = tem[:len_base]
        d_black_cbow.append(tem)
    d_white_cbow = []
    d_white_raw = []
    len_base = 32
    for sample in d_white:

        d_white_raw.append(sample['feature'])
    
    for sample in d_white_raw:
        tem = []
        for value in sample:
            tem.append(list(model.wv.get_vector(value)))
        while len(tem)<len_base:
            tem.append([0] *300)
        tem = tem[:len_base]
        d_white_cbow.append(tem)

    dataset_black_cbow = []
    for i in range(len(d_black)):
        sample = {}
        sample["feature"] = d_black_cbow[i]
        sample["name"] = d_black[i]["name"]
        sample["label"] = d_black[i]["label"]
        dataset_black_cbow.append(sample)
    d_black_cbow_np = np.array(dataset_black_cbow)

    dataset_white_cbow = []
    for i in range(len(d_white)):
        sample = {}
        sample["feature"] = d_white_cbow[i]
        sample["name"] = d_white[i]["name"]
        sample["label"] = d_white[i]["label"]
        dataset_white_cbow.append(sample)
    d_white_cbow_np = np.array(dataset_white_cbow)
    return d_black_cbow_np, d_white_cbow_np
# def pre_data_word():
#     from gensim.models import word2vec
#     import torch
#     import numpy as np 
#     d_black = np.load("data/datacon_black_tlsWord_ip.npy", allow_pickle=True)
#     d_white = np.load("data/datacon_white_tlsWord_ip.npy", allow_pickle=True)
#     tests = []
#     len_count = []
#     for dataset in [d_black, d_white]:
#         for data in dataset:
#             tests.append(data["feature"])
#             len_count.append(len(data["feature"]))
#     tem = np.array(len_count)
#     tem.mean()
#     from gensim.models import word2vec
#     model = word2vec.Word2Vec(tests, vector_size=300, min_count=1)
#     d_black_cbow = []
#     d_black_raw = []
#     for sample in d_black:

#         d_black_raw.append(sample['feature'])
    
#     for sample in d_black_raw:
#         tem = []
#         for value in sample:
#             tem.append(list(model.wv.get_vector(value)))
#         while len(tem)<17:
#             tem.append([0] *300)
#         tem = tem[:17]
#         d_black_cbow.append(tem)
#     d_white_cbow = []
#     d_white_raw = []
#     for sample in d_white:

#         d_white_raw.append(sample['feature'])
    
#     for sample in d_white_raw:
#         tem = []
#         for value in sample:
#             tem.append(list(model.wv.get_vector(value)))
#         while len(tem)<17:
#             tem.append([0] *300)
#         tem = tem[:17]
#         d_white_cbow.append(tem)

#     dataset_black_cbow = []
#     for i in range(len(d_black)):
#         sample = {}
#         sample["feature"] = d_black_cbow[i]
#         sample["name"] = d_black[i]["name"]
#         sample["label"] = d_black[i]["label"]
#         dataset_black_cbow.append(sample)
#     d_black_cbow_np = np.array(dataset_black_cbow)

#     dataset_white_cbow = []
#     for i in range(len(d_white)):
#         sample = {}
#         sample["feature"] = d_white_cbow[i]
#         sample["name"] = d_white[i]["name"]
#         sample["label"] = d_white[i]["label"]
#         dataset_white_cbow.append(sample)
#     d_white_cbow_np = np.array(dataset_white_cbow)
#     return d_black_cbow_np, d_white_cbow_np

    
def pre_data(dataset_name_attack, dataset_name_normal, feature_type, args):
    print("begin data_prepare")
    base_path_attack = "data/{}_{}.npy".format(dataset_name_attack, feature_type)
    base_path_normal = "data/{}_{}.npy".format(dataset_name_normal, feature_type)
    if args.f == "tlsWord":
        dataset_attack, dataset_normal = pre_tlsWord()
    else:
        dataset_attack = np.load(base_path_attack, allow_pickle=True)
        dataset_normal = np.load(base_path_normal, allow_pickle=True)
   
    num_attack = len(dataset_attack)
    num_normal = len(dataset_normal)
    num_min = min(num_attack, num_normal)
    
    random.shuffle(dataset_attack)
    random.shuffle(dataset_normal)

    if dataset_name_attack == "datacon_black" and dataset_name_normal == "datacon_white":
        num_all = num_normal + num_attack
        dataset_raw = np.hstack((dataset_attack, dataset_normal))
        num_test = min(int(num_all * 0.1), num_min)
    else:
        num_all = 2 * (num_min)
        dataset_raw = np.hstack((dataset_attack[:num_normal], dataset_normal))
        dataset_attack = []
        dataset_normal = []
        num_test = int(num_all * 0.2)
    
    print(num_attack, num_normal)
    print(num_test)
    
    dataset = []
    label = []
    name, name_test = [], []

    if feature_type == "word":
        num_1, num_2, num_3, len_1, len_2, len_3 = 0, 0, 0, len(dataset_raw[0]["client_hello"]), len(dataset_raw[0]["server_hello"]), len(dataset_raw[0]["certificate"])
    
    if feature_type == "flow_feature":
        subject = []
        issue = []
        alo = []
        ip_src = []
        ip_dst = []

    for (i, key) in enumerate(dataset_raw):
        if feature_type == "flow_feature":
            dataset.append(key["feature"])
            subject.append(key["subject"] if key["subject"] else "")
            issue.append(key["issue"] if key["issue"] else "")
            alo.append(key["signature_alo"] if key["signature_alo"] else "")
            ip_src.append(key["ip_src"])
            ip_dst.append(key["ip_dst"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "img" or feature_type == "img_noip":
            dataset.append(key["feature"])
            # dataset.append(key["feature"].flatten().reshape(-1, 1))
            label.append(0 if key["lable"] == "normal" or key["lable"] == "white" else 1)
        elif feature_type == "seq":
            size = 30
            # dataset.append(key["pay_seq"][:size])
            dataset.append(np.array(key["pay_seq"][:size]).reshape(-1, 1))
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "word":
            dataset.append(np.vstack((key["client_hello"], key["server_hello"], key["certificate"])))
            if key["client_hello"] == [[0, 0, 0, 0]] * len_1:
                num_1 += 1
            if key["server_hello"] == [[0, 0, 0, 0]] * len_2:
                num_2 += 1
            if key["certificate"] == [[0, 0, 0, 0]] * len_3:
                num_3 += 1
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "contrast_1":
            # dataset.append(key["feature"])
            dataset.append(key["feature"][:21])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "contrast_2":
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "content_seq":
            feature = []
            for i in range(3):
                feature.append(np.hstack(([key["nth_seq"][i]], key["pay_seq"][i], key["content_seq"][i])))
            dataset.append(feature)
            label.append(0 if key["lable"] == "normal" or key["lable"] == "white" else 1)
        elif feature_type == "mult_seq":
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "content_seq_2_ip" or feature_type == "content_seq_2":
            dataset.append(key["content_seq"])
            label.append(0 if key["lable"] == "normal" or key["lable"] == "white" else 1)
        elif feature_type == "mix_contrast_ip":
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "tlsWord_cbow":
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "tlsWord_skip":
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        elif feature_type == "anderson_":
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        # else:
        #     sample = np.array(key["feature"])
        #     dataset.append(sample[:args.word_num, 6-args.word_len:])
        #     label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
        else:
            dataset.append(key["feature"])
            label.append(0 if key["label"] == "normal" or key["label"] == "white" else 1)
            name.append(key["name"])
    if feature_type ==  "word":
        print("client_hello:{} server_hello:{} certifcate:{}".format(num_1, num_2, num_3))
    if feature_type == "flow_feature":
        one_subject = OneHotEncoder(sparse=False)
        one_issue = OneHotEncoder(sparse=False)
        one_alo = OneHotEncoder(sparse=False)
        one_ip_src = OneHotEncoder(sparse=False)
        one_ip_dst = OneHotEncoder(sparse=False)

        subject = one_subject.fit_transform(np.array(subject).reshape(-1, 1))
        issue = one_issue.fit_transform(np.array(issue).reshape(-1,1))
        alo = one_alo.fit_transform(np.array(alo).reshape(-1,1))
        ip_src = one_ip_src.fit_transform(np.array(ip_src).reshape(-1, 1))
        ip_dst = one_ip_dst.fit_transform(np.array(ip_dst).reshape(-1, 1))

        dataset = np.array(dataset)
        MinMax = MinMaxScaler(feature_range=(0, 1))
        dataset = MinMax.fit_transform(dataset)
        dataset = np.hstack((dataset, subject, issue, alo, ip_src, ip_dst))
        # dataset = np.hstack((dataset, subject, issue, alo))
        print(dataset.shape)

        Var = VarianceThreshold(threshold=0)
        dataset = Var.fit_transform(dataset)
    with open("white_all.txt", "w") as f :
        for key in name[:num_test] + name[-num_test:]:
            if "white" in key:
                f.write(key + "\n")
    # if dataset_name_attack == "datacon_black" and dataset_name_normal == "datacon_white":
    # if len(dataset[0]) > 2 * num_test:
    x, x_test, y, y_label = np.array(dataset[num_test:-num_test]), np.vstack((dataset[:num_test], dataset[-num_test:])), np.array(label[num_test:-num_test]), np.array(label[:num_test] + label[-num_test:])
        # if feature_type == "mix_word_seq_ip":
        #     name, name_test = np.array(name[num_test:-num_test]), np.vstack((name[:num_test], name[-num_test:]))
    # else:
    #     x, x_test, y, y_label = dataset[num_test:-num_test], dataset[:num_test] + dataset[-num_test:], label[num_test:-num_test], label[:num_test] + label[-num_test:]
        # if feature_type == "mix_word_seq_ip":
        #     name, name_test = name[num_test:-num_test], name[:num_test] + name[-num_test:]
    if dataset_name_attack == "datacon_black" and dataset_name_normal == "datacon_white":    
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, random_state= 66)
    elif dataset_name_attack == "malware-mta" and dataset_name_normal == "normal":
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, random_state= 66)
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, random_state= 66)
    print("end data_prepare")
    # if feature_type == "mix_word_seq_ip":
    #     return x_train, x_val, x_test, y_train, y_val, y_label
    # else:
    return x_train, x_val, x_test, y_train, y_val, y_label


def pre_dataset(args):
    # x, y = pre_data_cut()
    # if feature_type == "mix_word_seq_ip":
    #     x_train, x_val, x_test, y_train, y_val, y_label, name, name_test = pre_data(dataset_name_attack=d_attack, dataset_name_normal=d_normal, feature_type=feature_type)
    # else:
    d_attack, d_normal, feature_type, batch_size, args = args.d_1, args.d_2, args.f, args.b, args
    x_train, x_val, x_test, y_train, y_val, y_label = pre_data(dataset_name_attack=d_attack, dataset_name_normal=d_normal, feature_type=feature_type, args=args)

    if feature_type == "contrast_1" or feature_type =="content_seq" or feature_type == "contrast_2":
        x_train = torch.tensor(x_train, dtype=torch.int)
        x_test = torch.tensor(x_test, dtype=torch.int)
        x_val = torch.tensor(x_val, dtype=torch.int)
    else:
        x_train = torch.tensor(x_train, dtype=torch.float)
        x_test = torch.tensor(x_test, dtype=torch.float)
        x_val = torch.tensor(x_val, dtype=torch.float)
    y_label = torch.tensor(y_label)
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    dataloaders_dict = {
        'train': Data.DataLoader(Data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4),
        'val': Data.DataLoader(Data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4),
        'test': Data.DataLoader(Data.TensorDataset(x_test, y_label), batch_size=batch_size, shuffle=False, num_workers=4)
        }

    dataset_sizes = {'train': len(x_train),
                     'val': len(x_val),
                     'test': len(x_test)}
    print(dataset_sizes)
    # if feature_type == "mix_word_seq_ip":
    #     return dataloaders_dict, dataset_sizes, name, name_test

    return dataloaders_dict, dataset_sizes

if __name__ == "__main__":
    pre_dataset("datacon_black", "datacon_white", "content_seq", 32)