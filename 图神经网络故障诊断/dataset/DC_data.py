import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from dataset.__construct_graph import generate_graph

def data_preprocessing(dataset_path,sample_number,window_size,overlap,normalization,noise,snr,
                         input_type,graph_type,K,p,node_num,direction,edge_type,edge_norm,train_size,batch_size):

    root = dataset_path
    csv_name = ['train.csv', 'test_data.csv']

    data1 = [[], [], [], [], [], [], [], [], [], []]
    with open(os.path.join(root, csv_name[0]), encoding='gbk') as file:
        for line in file.readlines()[1:]:  # 从第二行开始读取数据
            line = line.split(',')[1:]  # 用逗号分隔数据，并舍弃第一列的编号id  6001 ---6000 data + 1label
            line = list(map(lambda x: float(x), line))  # 将6001个字符转化为数字
            data1[int(line[-1])].append(line[:-1])  # 按照标签将6000个数据存到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], []]
    for data1_index in range(len(data1)):
        data[data1_index].append(data1[data1_index][:43])

    data = np.array(data).squeeze(axis=1)  # shape: (10,43,6000)
    data = data[:, :, :(data.shape[2] // window_size) * window_size]  # shape: (10,43,5120)  when window_size == 1024

    data = data.reshape((data.shape[0], data.shape[1], data.shape[2] // window_size,
                         window_size))  # 将5120个数据按照window_size划分  shape: (10,43,5,1024)
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2], data.shape[3]))  # shape: (10,215,1024)
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))  # shape: (10,215*1024)

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