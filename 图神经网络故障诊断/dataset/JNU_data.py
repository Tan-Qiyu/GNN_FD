import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from dataset.__construct_graph import generate_graph

def data_preprocessing(dataset_path,sample_number,window_size,overlap,normalization,noise,snr,
                         input_type,graph_type,K,p,node_num,direction,edge_type,edge_norm,train_size,batch_size):

    root = dataset_path

    health = ['n600_3_2.csv', 'n800_3_2.csv', 'n1000_3_2.csv']  # 600 800 1000转速下的正常信号
    inner = ['ib600_2.csv', 'ib800_2.csv', 'ib1000_2.csv']  # 600 800 1000转速下的内圈故障信号
    outer = ['ob600_2.csv', 'ob800_2.csv', 'ob1000_2.csv']  # 600 800 1000转速下的外圈故障信号
    ball = ['tb600_2.csv', 'tb800_2.csv', 'tb1000_2.csv']  # 600 800 1000转速下的滚动体故障信号

    file_name = []  # 存放三种转速下、四种故障状态的文件名，一共12种类型
    file_name.extend(health)
    file_name.extend(inner)
    file_name.extend(outer)
    file_name.extend(ball)

    data1 = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据(每一类数据不平衡)
    for num, each_name in enumerate(file_name):
        dir = os.path.join(root, each_name)
        with open(dir, "r", encoding='gb18030', errors='ignore') as f:
            for line in f:
                line = float(line.strip('\n'))  # 删除每一行后的换行符号，并将字符型转化为数字
                data1[num].append(line)  # 将取出来的数据逐个存放到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据（每一类数据平衡）shape：(12,500500)
    for data1_i in range(len(data1)):
        data[data1_i].append(data1[data1_i][:500500])  # 将所有类型数据总长度截取为500500

    data = np.array(data).squeeze(axis=1)  # shape：(12,500500)

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // window_size, window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,
                                                          overlap=overlap)

    sample_data = sample_data[:, :sample_number, :]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    if input_type == 'TD':  #时域信号
        data = norm_data
    elif input_type == 'FD':  #频域信号
        data = np.zeros((norm_data.shape[0],norm_data.shape[1],norm_data.shape[2]))
        for label_index in range(norm_data.shape[0]):
            fft_data = FFT(norm_data[label_index,:,:])
            data[label_index,:,:] = fft_data

    graph_dataset = generate_graph(feature=data,graph_type=graph_type,node_num=node_num,direction=direction,
                                   edge_type=edge_type,edge_norm=edge_norm,K=K,p=p)

    str_y_1 = []
    for i in range(len(graph_dataset)):
        str_y_1.append(np.array(graph_dataset[i].y))

    train_data, test_data = train_test_split(graph_dataset, train_size=train_size, shuffle=True,stratify=str_y_1)  # 训练集、测试集划分

    loader_train = DataLoader(train_data,batch_size=batch_size)
    loader_test = DataLoader(test_data,batch_size=batch_size)

    return loader_train, loader_test