import math
import numpy as np

#数据归一化
def Normal_signal(data,normal_type):
    '''
    :param data: the data before normalization ---- shape:(samples,windows size)---egg:(100,1024)
    :param normal_type: the method of normalization
    :return data_norm : the data after normalization ---- shape:(samples,windows size)---egg:(100,1024)
    '''
    num, len = data.shape[0], data.shape[1]
    data_norm = np.zeros((num,len))  #创建一个全为0矩阵存放归一化后的数据
    for i in range(num):
        if normal_type == 'Z-score Normalization':
            mean, var = data[i].mean(axis=0), data[i].var(axis=0)
            data_norm[i] =(data[i] - mean) / np.sqrt(var)
            data_norm[i] =data_norm[i].astype("float32")

        elif normal_type == 'Max-Min Normalization':
            maxvalue, minvalue = data[i].max(axis=0), data[i].min(axis=0)
            data_norm[i] = (data[i] - minvalue) / (maxvalue - minvalue)
            data_norm[i] = data_norm[i].astype("float32")

        elif normal_type == '-1 1 Normalization':
            maxvalue, minvalue = data[i].max(axis=0), data[i].min(axis=0)
            data_norm[i] = -1 + 2 * ((data[i] - minvalue) / (maxvalue - minvalue))
            data_norm[i] = data_norm[i].astype("float32")

        else:
            print('the normalization is not existed!!!')
            break

    return data_norm

#定义滑窗采样及其样本重叠数
def Slide_window_sampling(data, window_size,overlap):
    '''
    :param data: the data raw signals with length n
    :param window_size: the sampling length of each samples
    :param overlap: the data shift length of neibor two samples
    :return squence: the data after sampling
    '''

    count = 0  # 初始化计数器
    data_length = int(data.shape[0])  # 信号长度
    sample_num = math.floor((data_length - window_size) / overlap + 1)  # 该输入信号得到的样本个数
    squence = np.zeros((sample_num, window_size), dtype=np.float32)  # 初始化样本数组
    for i in range(sample_num):
        squence[count] = data[overlap * i: window_size + overlap * i].T  # 更新样本数组
        count += 1  # 更新计数器

    return squence

#添加不同信噪比噪声
def Add_noise(x, snr):
    '''
    :param x: the raw sinal after sliding windows sampling
    :param snr: the snr of noise
    :return noise_signal: the data which are added snr noise
    '''
    d = np.random.randn(len(x))  # generate random noise
    P_signal = np.sum(abs(x) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (snr / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = x.reshape(-1) + noise
    return noise_signal


#傅里叶变换
def FFT(x):
    '''
    :param x: time frequency signal
    :return y: frequency signal
    '''
    #y = np.empty((x.shape[0],int(x.shape[1] / 2)))  #单边频谱
    y = np.empty((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        y[i] = (np.abs(np.fft.fft(x[i])) / len(x[i]))   #傅里叶变换、取幅值、归一化
        #y[i] = (np.abs(np.fft.fft(x[i])) / len(x[i]))[range(int(x[i].shape[0] / 2))]  # 傅里叶变换、取幅值、归一化、取单边
    return y