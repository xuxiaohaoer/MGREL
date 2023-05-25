import numpy as np 
import torch 
import random
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import sklearn



import sys
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
from pre_data import pre_data
def RandomForest(argv):
    """
    dataset_1:第一个数据集
    dataset_2:第二个数据集
    feature_type:特征类型
    n_iter:迭代次数
    """
    dataset_1 = argv.d_1
    dataset_2 = argv.d_2
    feature_type = argv.f
    n_iter = 0
    x_train, x_val, x_test, y_train, y_val, y_label = pre_data(dataset_1, dataset_2, feature_type, args)
    print("begin model")
    rnd_clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
    rnd_clf.fit(x_train, y_train)
    y_pred = rnd_clf.predict(x_test)
  
    print("acc:{:.4f}".format(accuracy_score(y_label, y_pred)))
    print("pre:{:.4f}".format(precision_score(y_label, y_pred)))
    print("rec:{:.4f}".format(recall_score(y_label, y_pred)))
    print("f1:{:.4f}".format(f1_score(y_label, y_pred)))
    y_proba = rnd_clf.predict_proba(x_test)[:,1]
    print("auc:{:.4f}".format(roc_auc_score(y_label, y_proba)))
    print(confusion_matrix(y_label, y_pred, labels=[1, 0]))

    acc, pre, rec, f1, auc = accuracy_score(y_label, y_pred), precision_score(y_label, y_pred), recall_score(y_label, y_pred), f1_score(y_label, y_pred), roc_auc_score(y_label, y_proba)
    with open('lr_ill.txt', 'a+',encoding='utf-8') as f:
        list = []
        list.append("acc:{:.4f}\n".format(acc))
        list.append("pre:{:.4f}\n".format(pre))
        list.append("rec:{:.4f}\n".format(rec))
        list.append("f1:{:.4f}\n".format(f1))
        list.append("auc:{:.4f}\n".format(auc))
        list.append("matrix:{}\n".format(confusion_matrix(y_label, y_pred)))

        f.writelines(list)
    print("end model")
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_1", type=str, default="malware-mta", help="first dataset")
    parser.add_argument("--d_2", type=str, default="normal", help="second dataset")
    parser.add_argument("--f", type=str, default="anderson", help="select feature")
    args = parser.parse_args()
    RandomForest(args)
