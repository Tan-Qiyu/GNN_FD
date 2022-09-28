from scipy.io import loadmat
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from dataset.__construct_graph import generate_graph

'''Normal Baseline Data由于仅有4个文件97.mat、98.mat、99.mat、100.mat，
为了方便访问及处理，将四个文件都拷贝到其余三个文件夹中'''

#正常数据，负载0、1、2、3
NB = ['97.mat', '98.mat', '99.mat', '100.mat']

'''12k Drive End Bearing Fault Data'''
#内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_12DE = ['105.mat', '106.mat', '107.mat', '108.mat']
IR14_12DE = ['169.mat', '170.mat', '171.mat', '172.mat']
IR21_12DE = ['209.mat', '210.mat', '211.mat', '212.mat']

#外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_12DE = ['130.mat', '131.mat', '132.mat', '133.mat']
OR14_12DE = ['197.mat', '198.mat', '199.mat', '200.mat']
OR21_12DE = ['234.mat', '235.mat', '236.mat', '237.mat']

#滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_12DE = ['118.mat', '119.mat', '120.mat', '121.mat']
B14_12DE = ['185.mat', '186.mat', '187.mat', '188.mat']
B21_12DE = ['222.mat', '223.mat', '224.mat', '225.mat']

#全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_12DE = [NB,  IR07_12DE, IR14_12DE, IR21_12DE, OR07_12DE, OR14_12DE, OR21_12DE,B07_12DE, B14_12DE, B21_12DE]

'''48k Drive End Bearing Fault Data'''
#内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_48DE = ['109.mat', '110.mat', '111.mat', '112.mat']
IR14_48DE = ['174.mat', '175.mat', '176.mat', '177.mat']
IR21_48DE = ['213.mat', '214.mat', '215.mat', '217.mat']

#外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_48DE = ['135.mat', '136.mat', '137.mat', '138.mat']
OR14_48DE = ['201.mat', '202.mat', '203.mat', '204.mat']
OR21_48DE = ['238.mat', '239.mat', '240.mat', '241.mat']

#滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_48DE = ['122.mat', '123.mat', '124.mat', '125.mat']
B14_48DE = ['189.mat', '190.mat', '191.mat', '192.mat']
B21_48DE = ['226.mat', '227.mat', '228.mat', '229.mat']

#全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_48DE = [NB, IR07_48DE, IR14_48DE, IR21_48DE, OR07_48DE, OR14_48DE, OR21_48DE,B07_48DE, B14_48DE, B21_48DE]
full_data = [full_data_12DE,full_data_48DE]

# ================================the processing of train data and test data ========================================
def data_preprocessing(dataset_path,sample_number,dir_path,window_size,overlap,normalization,noise,snr,
                         input_type,graph_type,K,p,node_num,direction,edge_type,edge_norm,train_size,batch_size):
    '''
    :param dataset_path: the file path of cwru datasets
    :param sample_number: the samples numbers of each fault type and 4 motor rpm ----total samples = samples_number * 40 ---- 40 = 10(fault classes) * 4(4 motor rpm)
    :param train_size: train sample / totlal samples
    :param dir_path: the type of vibration sensors signal(different sample frequency)  ---- 12DE: 12k Drive End Bearing Fault Data; 48DE: 48k Drive End Bearing Fault Data
    :param Window_size: the sample length of each sample
    :param overlap: the data shift of neibor two samples
    :param normalization: the type of normalization
    :param noise: add noise or don't add noise
    :param snr: the snr of noise
    :param input_type: time domain or frequency domain signal as input(node feature)
    :param graph_type: the type of constructed graph
    :param K: the number of neibor in knn graph
    :param peason_threshold_value: the connected threshold value of each two nodes in peason graph
    :param p: the connected proablity of each two nodes in ER random graph
    :param node_num: the number of a graph
    :param direction: directed graph or undirected graph
    :param edge_type: the weight method of edge
    :param  edge_norm: whether normalize edge weight
    :param batch_size:
    :return loader_train, loader_test: 图训练集、图测试集
    '''

    File_dir = dataset_path

    dirname = ['Normal Baseline Data','12k Drive End Bearing Fault Data','48k Drive End Bearing Fault Data']

    if dir_path == '12DE':
        data_path = os.path.join(File_dir,dirname[1])
        file_number = 0
    elif dir_path == '48DE':
        data_path = os.path.join(File_dir, dirname[2])
        file_number = 1

    data_list = [[],[],[],[],[],[],[],[],[],[]]

    i = 0

    for bearing_state in enumerate(full_data[file_number]):

        for num,load in enumerate(bearing_state[1]):
            data = loadmat(os.path.join(data_path, load))
            if eval(load.split('.')[0]) < 100:
                vibration = data['X0' + load.split('.')[0] + '_DE_time']
            elif eval(load.split('.')[0]) == 174:
                vibration = data['X' + '173' + '_DE_time']
            else:
                vibration = data['X' + load.split('.')[0] + '_DE_time']

            #添加不同信噪比的噪声
            if noise == 'y' or noise == 1:
                vibration = Add_noise(vibration,snr).reshape(Add_noise(vibration,snr).shape[0],1)
            elif noise == 'n' or noise == 0:
                vibration = vibration

            slide_data = Slide_window_sampling(vibration, window_size, overlap)  # 滑窗采样
            if normalization != 'unnormalization':
                data_x = Normal_signal(slide_data,normalization)  #归一化
            else:
                data_x = slide_data

            # np.random.shuffle(data_x)  #将数据shuffle

            if sample_number == 112 and num == 0:  #当每种故障下不同负载的样本数为112，总样本数为112*40=4480时，在每种故障下的0负载增加2个样本数使得样本总数为4480+2*10=4500
                data_sample_x = data_x[:sample_number + 2,:]  #时域信号
                data_list[i].append(data_sample_x)
            else:
                data_sample_x = data_x[:sample_number, :]  # 时域信号
                data_list[i].append(data_sample_x)

        i = i + 1

    all_data = []

    for sample in data_list:
        each_data = np.concatenate((sample[0],sample[1],sample[2],sample[3]),axis=0)
        all_data.append(each_data)

    #当每一类故障的样本不平衡时截取相同数量的样本
    sam_list = []
    list(map(lambda x:sam_list.append(x.shape[0]),all_data))
    sam_min = min(sam_list)
    max_min = max(sam_list)
    if sam_min != max_min:
        balance_data = [[],[],[],[],[],[],[],[],[],[]]
        for all_data_index,class_data in enumerate(all_data):
            #np.random.shuffle(class_data)
            balance_data[all_data_index].append(all_data[all_data_index][:sam_min,:])
        all_data = np.array(balance_data).squeeze(axis=1)

    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  #numpy版本过高，添加此代码可忽略警告
    all_data = np.array(all_data)  #shape: (10,samples num,Window_size) --- egg : (10,450,1024)

    #对每一个样本进行FFT, data -- shape (10，450，1024)  10-故障类型数；450-每种故障样本数；1024-每个样本长度
    if input_type == 'TD':  #时域信号
        data = all_data
    elif input_type == 'FD':  #频域信号
        data = np.zeros((all_data.shape[0],all_data.shape[1],all_data.shape[2]))
        for label_index in range(all_data.shape[0]):
            fft_data = FFT(all_data[label_index,:,:])
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