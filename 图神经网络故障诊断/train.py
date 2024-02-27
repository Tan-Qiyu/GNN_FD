'''
Time: 2022/9/28 23:46
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
Dataset Download Link：https://github.com/Tan-Qiyu/Mechanical_Fault_Diagnosis_Dataset
引用格式:[1]谭启瑜,马萍,张宏立.基于图卷积神经网络的滚动轴承故障诊断[J].噪声与振动控制,2023,43(06):101-108+116.
'''

'''
--------------------------------------------------参数介绍------------------------------------------
dataset_name： 数据集名称
    CWRU、SEU、XJTU、JNU、MFPT、UoC、DC 共7个公开数据集
    
dataset_path： 数据集目录地址
    CWRU  "E:\故障诊断数据集\凯斯西储大学数据" -------------------------------------------# 凯斯西储大学轴承数据集
    SEU   "E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\bearingset" --------- -# 东南大学轴承子数据
    SEU   "E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\gearset" --------------# 东南大学齿轮子数据
    XJTU  "E:\故障诊断数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets"----# 西安交通大学-昇阳轴承退化数据集
    JNU   "E:\故障诊断数据集\江南大学轴承数据(一)\数据"-------------------------------------# 江南大学轴承数据集
    MFPT  "E:\故障诊断数据集\MFPT Fault Data Sets"--------------------------------------# 美国-机械故障预防技术学会MFPT数据集
    UoC   "E:\故障诊断数据集\美国-康涅狄格大学-齿轮数据集"------------------------------------# 美国-康涅狄格大学-齿轮数据集
    DC    "E:\故障诊断数据集\DC轴承数据"--------------------------------------------------# 中国-轴承数据集（DC竞赛）

dir_path:   CWRU数据集的采样频率和传感器位置
    12DE  # 12kHZ Drive End Dataset
    48DE  # 48kHZ Drive End Dataset

SEU_channel:  SEU数据集的数据通道
    0、1、2、3、4、5、6、7  共8个通道

minute_value：  XJTU-SY数据集使用最后多少个文件数据进行实验验证

XJTU_channel：   XJTU-SY数据集的数据通道
    X Y XY  共3种通道
    
sample_num：   每一种故障类型的样本数（CWRU数据集除外，为每一种故障下的一种工况的样本数，即CWRU每一类故障样本数为sample_num * 工况数 = sample_num * 4）

train_size： 训练集比例

sample_length：  样本采样长度 = 网络的输入特征长度

overlap：   滑窗采样偏移量，当sample_length = overlap时为无重叠顺序采样

norm_type：  原始振动信号的归一化方式
    unnormalization        # 不进行归一化
    Z-score Normalization  # 均值方差归一化
    Max-Min Normalization  # 最大-最小归一化：归一化到0到1之间
    -1 1 Normalization     # 归一化到-1到1之间

noise:   是否往原始振动信号添加噪声
    0   # 不添加噪声
    1   # 添加噪声

snr：   当noise = 1时添加的噪声的信噪比大小；当noise = 0时此参数无效

input_type：  输入信号的类型
    TD   # 时域信号作为特征输入
    FD   # 频域信号（FFT之后的双边归一化频谱）作为特征输入
   
graph_type：  振动信号构造的图类型 
    path_graph       # 路图
    knn_graph        # K近邻图
    complete_graph   # 完全图
    ER_graphER       # ER随机图
    visibility_graph # 可视图

knn_K：  K近邻图的k值

ER_p：  ER随机图的任意两节点的连接概率值

node_num：  每个图的节点个数

direction:  图的有向性
    directed       # 有向图
    undirected     # 无向图

edge_type：  边加权方式  
    '0-1'  #0-1等值加权
    'Euclidean Distance'  # 欧几里得距离
    'Manhattan Distance'  # 曼哈顿距离
    'Chebyshev Distance'  # 切比雪夫距离
    'Minkowski Distance'  # 闵可夫斯基距离
    'Hamming Distance'    # 汉明距离
    'Cosine Distance'     # 余弦相似度
    'Pearson Correlation Coefficient'  # 皮尔逊相关系数
    'Jaccard Similarity Coefficient'   #杰卡德相似系数
    'Gaussian kernel'    # 高斯核函数加权
    
edge_norm： 是否将加权边归一化
    True     # 归一化
    False    # 不归一化
    
batch_size:   批处理大小

model_type:  网络模型
    MLP、LeNet、1D-CNN、ChebyNet、GCN、SGCN、GAT、GIN、GraphSAGE 

epochs：  迭代轮数

learning_rate：  学习率

momentum：  动量因子

optimizer:  优化器

visualization:  是否绘制混淆矩阵与每一层网络的t-SNE可视化图
    True   # 进行可视化
    False  # 不进行可视化
'''

import argparse
from utils.train_utils import train_utils

def parse_args():
    parser = argparse.ArgumentParser()
    # basic parameters
    #===================================================dataset parameters=============================================================================
    parser.add_argument('--dataset_name', type=str, default='CWRU', help='the name of the dataset：CWRU、SEU、XJTU、JNU、MFPT、UoC、DC')
    parser.add_argument('--dataset_path', type=str, default=r"E:\故障诊断数据集\凯斯西储大学数据", help='the file path of the dataset')
    parser.add_argument('--dir_path', type=str, default='12DE', help='the sample frequency of CWRU：12DE、48DE represent 12kHZ and 48kHZ respectively')
    parser.add_argument('--SEU_channel', type=int, default=1, help='the channel number of SEU：0-7')
    parser.add_argument('--minute_value', type=int, default=10, help='the last (minute_value) csv file of XJTU datasets each fault class')
    parser.add_argument('--XJTU_channel', type=str, default='X', help='XJTU channel signal:X 、Y 、XY')

    # ===================================================data preprocessing parameters=============================================================================
    parser.add_argument('--sample_num', type=int, default=50,help='the number of samples')
    parser.add_argument('--train_size', type=float, default=0.6, help='train size')
    parser.add_argument('--sample_length', type=int, default=1024, help='the length of each samples')
    parser.add_argument('--overlap', type=int, default=1024, help='the sampling shift of neibor two samples')
    parser.add_argument('--norm_type', type=str, default='unnormalization',help='the normlized method')
    parser.add_argument('--noise', type=int, default=0, help='whether add noise')
    parser.add_argument('--snr', type=int, default=0, help='the snr of noise')

    parser.add_argument('--input_type', type=str, default='FD',help='TD——time domain signal，FD——frequency domain signal')
    parser.add_argument('--graph_type', type=str, default='path_graph', help='the type of graph')
    parser.add_argument('--knn_K', type=int, default=5, help='the K value of knn-graph')
    parser.add_argument('--ER_p', type=float, default=0.5, help='the p value of ER-graph')
    parser.add_argument('--node_num', type=int, default=10, help='the number of node in a graph')
    parser.add_argument('--direction', type=str, default='undirected', help='directed、undirected')
    parser.add_argument('--edge_type', type=str, default='0-1', help='the edge weight method of graph')
    parser.add_argument('--edge_norm', type=bool, default=False, help='whether normalize edge weight')
    parser.add_argument('--batch_size', type=int, default=64)

    # ===================================================model parameters=============================================================================
    parser.add_argument('--model_type', type=str, default='GCN', help='the model of training and testing')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')

    # ===================================================visualization parameters=============================================================================
    parser.add_argument('--visualization', type=bool, default=True, help='whether visualize')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    train_utils(args)

