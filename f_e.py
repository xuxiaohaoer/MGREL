import os
import tqdm
from feature_extract.flow import Flow
# from flow import Flow
import dpkt
import numpy as np
from tqdm import tqdm
# malware-lastline malware-mta malware-strtosphere normal datacon CTU-13
import sys
import argparse

def pre_flow(args):
    dataset_name = args.d
    feature_type = args.f
    # feature_type = "flow_feature"
    print("dataset:{} feature_type:{}".format(dataset_name, feature_type))
    data_path = "data_flow/{}/tls/".format(dataset_name)
    ip_flag = "" if args.ip else "no"
    print(ip_flag, args.ip, args.s, args.app)
    if args.ip:
        save_path = "data/{}_{}.npy".format(dataset_name, feature_type)
    else:
        save_path = "data/{}_{}_noip.npy".format(dataset_name, feature_type)
    
    ip_flag = "" if args.ip else "no"
    print(ip_flag, args.ip, args.s, args.app)
    # data_path = "../../data_flow/{}/tls/".format("datacon/train_white")
    # save_path = "../../data/{}_{}.npy".format("datacon_white", feature_type)

    dataset = []
    for filename in tqdm(os.listdir(data_path)):
        if ".pcap" in filename:
            try:
                with open(data_path + filename, 'rb') as f:
                    capture = dpkt.pcap.Reader(f)
                    type = filename.replace('.pcap', '').split("#")[1]
                    flow_sample = Flow(capture, type, args)
                    flow_sample.name = filename.replace('.pcap', '')
                    flow_sample.analyse()
                    if feature_type == "flow_feature":
                        feature = flow_sample.tolist()
                    elif feature_type == "img":
                        feature = flow_sample.toImag()
                    elif feature_type == "word":
                        feature = flow_sample.toWord()
                    elif feature_type == "content_seq":
                        feature = flow_sample.toConSeq()
                    elif feature_type == "seq":
                        feature = flow_sample.toSeq()
                    elif feature_type == "contrast_1":
                        feature = flow_sample.toContrast_1()
                    elif feature_type == "contrast_2":
                        feature = flow_sample.toContrast_2()
                    elif feature_type == "mult_seq":
                        feature = flow_sample.toMultSeq()
                    elif feature_type == "mult_dir_seq":
                        feature = flow_sample.toMultDirSeq()
                    elif feature_type == 'word_seq':
                        feature = flow_sample.toWordSeq(args.l)
                    elif feature_type == 'mix_word_seq':
                        feature = flow_sample.toMixWordSeq(args.l)
                    elif feature_type == 'mix_word_seq_pay':
                        feature= flow_sample.toMixWordSeq_2(args.l)
                    elif feature_type == 'mix_word_seq_1':
                        feature = flow_sample.toMixWordSeq(args.l)
                    elif feature_type == "word_seq_1":
                        feature = flow_sample.toWordSeq(args.l)
                    elif feature_type == "content_seq_2":
                        feature = flow_sample.toCon2Seq()
                    elif feature_type == 'mix_contrast':
                        feature = flow_sample.toMixContrast()    
                    elif feature_type == 'tlsWord':
                        feature = flow_sample.toTlsWord()
                    elif feature_type == 'tlsWord_raw':
                        feature = flow_sample.toTlsWord_raw()
                    elif feature_type == 'anderson':
                        feature = flow_sample.toAnderson()
                    dataset.append(feature)
                f.close()
            except IOError:
                print('could not parse {0}'.format(filename))
    
    dataset_np = np.array(dataset)
    if args.s:
        np.save(save_path, dataset_np)




if __name__ == "__main__":
    print("begin")
    parse = argparse.ArgumentParser()
    parse.add_argument("--d", type=str, default="normal", help="dataset")
    parse.add_argument("--f", type=str, default="mix_word_seq", help="feature_type")
    parse.add_argument("--s", type=bool, default=True, help="save or not")
    parse.add_argument("--m", type=str, default="datacon", help="dataset model")
    parse.add_argument("--l", type=int, default=95, help="word_num")
    parse.add_argument("--ip", type=bool, default=True)
    parse.add_argument("--tcp", type=bool, default=True)
    parse.add_argument("--app", type=bool, default=False)

    args = parse.parse_args()
    # pre_flow(testdir + "data_train/", '{}/train_packet_fre.npy'.format(path), '0')
    
    pre_flow(args)

    # pre_flow(sys.argv)
    # pre_flow()
    print("end")