# MGREL
本项目提供从特征提取到测试的模型全过程
## data_flow
* 读取原始流数据的默认根路径
* 请以`data_flow/xxx/tls`存放想要处理的xxx数据集
* 上述文件夹中，请存放按照五元组划分好的pcap文件
## data
* 读取提取特征的默认根路径
* 文件以`dataset_feature`命名，其中`dataset`为数据集名称，`feature`为所选择提取的特征
* 文件保存格式为`npy`
## feature_extract
* 本文件夹存放特征提取所用到的模块代码
## model
* 本文件夹存放各模型代码
* deepMal-model
* lstm-model
* MGREL-model（命名为mix_mult_bilstm及变体）
* M1CNN-model
* M2CNN-model
* RTETC-model
* TLSVEC-model
* MaIDIST-model
* DISSTILLER-model
* LR-LR
* RF-RF
## model_save
* 本文件夹提供训练好模型
## 使用说明
* `python f_e.py --d dataset`，启动特征提取文件，其中`dataset`为目标数据集
* `python ds.py`, 启动模型训练模块，模型保存在`model_save`，会根据所给标签输出测试指标
* --d_1 选择你的第一个数据集
* --d_2 选择你的第二个数据集
* --f 选择你的特征
* --m 选择你的模型
* --e 选择epoch
* --b 选择batch_size的大小
## Contributors
@xuxiaohaoer
## 其他
