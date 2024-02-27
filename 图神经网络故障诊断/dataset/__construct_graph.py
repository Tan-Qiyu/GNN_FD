import numpy as np
import torch
import networkx as nx
import scipy.spatial.distance as dist
from torch_geometric.data import Data
from torch_cluster import knn_graph

def edge_weight(x,edge_index,distance_type,edge_norm):  #边加权方式
    '''
    :param x: 每个图的节点特征
    :param edge_index: 连边信息
    :param distace_type: 边加权度量方式
    :return:
    '''
    if distance_type == '0-1':
        edge_attr = np.ones(edge_index.shape[1])
        return edge_attr
    else:
        edge_index, x = np.array(edge_index), np.array(x)
        edge_attr = np.empty(edge_index.shape[1])
        for edge_num in range(edge_index.shape[1]):
            source_node, target_node = edge_index[0][edge_num], edge_index[1][edge_num]  #取出源节点与目标节点编号
            source_node_feature, target_node_feature = x[source_node], x[target_node]   #取出源节点与目标节点特征
            if distance_type == 'Euclidean Distance':   #欧几里得距离
                distance = np.sqrt(np.sum(np.square(source_node_feature - target_node_feature)))
            elif distance_type == 'Manhattan Distance':  # 曼哈顿距离
                distance = np.sum(np.abs(source_node_feature - target_node_feature))
            elif distance_type == 'Chebyshev Distance':  # 切比雪夫距离
                distance =  np.abs(source_node_feature - target_node_feature).max()
            elif distance_type == 'Minkowski Distance':  # 闵可夫斯基距离
                distance = np.linalg.norm(source_node_feature - target_node_feature, ord=1)  #ord范数
            elif distance_type == 'Hamming Distance':  # 汉明距离
                distance = np.shape(np.nonzero(source_node_feature - target_node_feature)[0])[0]
            elif distance_type == 'Cosine Distance':  # 余弦相似度
                distance = np.dot(source_node_feature,target_node_feature)/(np.linalg.norm(source_node_feature)*(np.linalg.norm(target_node_feature)))
            elif distance_type == 'Pearson Correlation Coefficient':  # 皮尔逊相关系数
                distance = np.corrcoef(source_node_feature, target_node_feature)  #皮尔森相关系数矩阵
                distance = np.abs(distance[0,1])
            elif distance_type == 'Jaccard Similarity Coefficient':  #杰卡德相似系数,1
                distance = dist.pdist(np.array([source_node_feature, target_node_feature]), "jaccard")
            elif distance_type == 'Gaussian kernel':
                beta = 0.01
                Euclidean_distance = np.sqrt(np.sum(np.square(source_node_feature - target_node_feature)))
                distance = np.exp(-Euclidean_distance/(2*beta**beta))

            edge_attr[edge_num] = distance

        if edge_norm == True:
            edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())  #归一化
        return edge_attr

#路图
def path_graph(data,direction,edge_type,label,edge_norm):
    '''
    :param data: 每个图的节点特征
    :param direction: 有向图、无向图
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :return: path graph
    '''
    x = torch.tensor(data,dtype=torch.float) #节点特征
    if direction == 'directed':  #有向图
        edge_index = torch.tensor(np.array(nx.path_graph(data.shape[0]).edges).T, dtype=torch.long)
    elif direction == 'undirected':  # 无向图
        edge_index = torch.tensor(np.concatenate((np.array(nx.path_graph(data.shape[0]).edges).T,
                        np.roll(np.array(nx.path_graph(data.shape[0]).edges).T, shift=1, axis=0)),axis=1), dtype=torch.long)

    edge_attr = edge_weight(x=x,edge_index=edge_index,distance_type=edge_type,edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  #边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  #图

    return graph

#K-近邻图 -- 有向图
def Knn_graph(data,edge_type,label,edge_norm,K):
    '''
    :param data: 每个图的节点特征
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :param K：邻居数
    :return: knn graph
    '''
    data = (data[:, :, 1] if data.shape[-1] == 2 else data)  # 若取时域+频域信号，则计算频域信号的皮尔森相关系数进行GNN
    x = torch.tensor(data, dtype=torch.float)  # 节点特征
    batch = torch.tensor(np.repeat(0,data.shape[0]), dtype=torch.int64)
    edge_index = knn_graph(x, k=K, batch=batch, loop=False, flow='target_to_source')  # K为邻居数，此处为有向图
    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)  # 边加权方式
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

#全连接图
def complete_graph(data,edge_type,label,edge_norm):
    '''
    :param data: 每个图的节点特征
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :return: complete graph
    '''
    x = torch.tensor(data, dtype=torch.float)  # 节点特征
    edge_index = []
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

#ER随机图--无向图
def ER_graph(data,edge_type,label,edge_norm,p):
    '''
    :param data: 每个图的节点特征
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :param p: 任意两节点连接的概率值
    :return: ER random graph
    '''
    x = torch.tensor(data, dtype=torch.float)  # 节点特征
    edge_index = np.array(nx.random_graphs.erdos_renyi_graph(data.shape[0],p).edges).T
    edge_index = np.concatenate((edge_index, np.roll(edge_index, shift=1, axis=0)),axis=1)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

#可视图
def Visibility_graph(data,direction,edge_type,label,edge_norm):
    '''
    :param data: 每个图的节点特征
    :param direction:有向图、无向图
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :return: visibility graph
    '''
    data = (data[:, :, 1] if data.shape[-1] == 2 else data)  # 若取时域+频域信号，则计算频域信号的皮尔森相关系数进行GNN
    x = torch.tensor(data, dtype=torch.float)  # 节点特征

    series = torch.sum(x,dim=1)  #将节点特征求和作为可视图节点的幅值
    G = visibility_graph(series)  #返回graph
    edge_index = G.edges()  #此处取出的edge_index为有向边，即不满足对称性
    edge_index = list(edge_index)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    if direction == 'undirected':
        edge_index = torch.cat((edge_index, edge_index.roll(1, dims=0)), dim=1)  # 构建无向边

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

def generate_graph(feature,graph_type,node_num,direction,edge_type,edge_norm,K,p):
    '''
    :param feature: shape (classes，sample_num，sample_length)  classes-故障类型数；sample_num-每种故障样本数；sample_length-每个样本长度
    :param graph_type: 图类型
    :param node_num: 每个图的节点个数
    :param direction: 有向图、无向图
    :param edge_type: 边加权方式
    :param edge_norm: 边权重归一化
    :param K: knn graph的邻居数
    :param p: ER random graph的任意两节点的概率
    :return graph_dataset: 图数据集 -- 列表(故障类型数，图个数)
    '''

    graph_dataset = []  #按照故障类型逐次将图数据存入空列表中

    for label, class_fea in enumerate(feature):
        # np.random.shuffle(class_fea)  #将取出来的每一类故障信号shuffle
        start = 0
        end = node_num

        while end <= class_fea.shape[0]:
            a_graph_fea = class_fea[start:end,:]  #每一个图的节点特征

            if graph_type == 'path_graph':
                graph = path_graph(data=a_graph_fea,direction=direction,edge_type=edge_type,label=label,edge_norm=edge_norm)
            elif graph_type == 'knn_graph':
                graph = Knn_graph(data=a_graph_fea,edge_type=edge_type,label=label,edge_norm=edge_norm,K=K)
            elif graph_type == 'complete_graph':
                graph = complete_graph(data=a_graph_fea,edge_type=edge_type,label=label,edge_norm=edge_norm)
            elif graph_type == 'ER_graph':
                graph = ER_graph(data=a_graph_fea,edge_type=edge_type,label=label,edge_norm=edge_norm,p=p)
            elif graph_type == 'visibility_graph':
                graph = Visibility_graph(data=a_graph_fea, edge_type=edge_type, label=label, edge_norm=edge_norm,direction=direction)                
            else:
                print('this graph is not existed!!!')

            start = start + node_num
            end = end + node_num

            graph_dataset.append(graph)

    return graph_dataset


