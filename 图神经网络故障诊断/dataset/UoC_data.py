import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from dataset.__construct_graph import generate_graph

def data_preprocessing(dataset_path,sample_number,window_size,overlap,normalization,noise,snr,
                         input_type,graph_type,K,p,node_num,direction,edge_type,edge_norm,train_size,batch_size):

    root = dataset_path

    mat_name = ['DataForClassification_Stage0.mat', 'DataForClassification_TimeDomain.mat']
    file = loadmat(os.path.join(root, mat_name[1]))

    '''
    Number of gear fault types=9={'healthy','missing','crack','spall','chip5a','chip4a','chip3a','chip2a','chip1a'}
    Number of samples per type=104
    Number of total samples=9x104=903
    The data are collected in sequence, the first 104 samples are healthy, 105th ~208th samples are missing, and etc.
    '''
    all_data = file['AccTimeDomain']  # shape: (3600,936) --- 104*9 = 936

    each_class_num = 104
    start = 0
    end = each_class_num

    all_data = all_data.T  # shape: (936,3600)

    data = [[], [], [], [], [], [], [], [], []]  # 创建一个空列表存放九种故障信号
    data_index = 0
    while end <= all_data.shape[0]:
        data[data_index].append(all_data[start:end])
        start += each_class_num
        end += each_class_num
        data_index += 1

    data = np.array(data).squeeze(axis=1)  # shape: (9,104,3600)
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])  # shape: (9,374400)

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
