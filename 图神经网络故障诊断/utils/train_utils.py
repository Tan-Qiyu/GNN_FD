import torch
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import CWRU_data,SEU_data,XJTU_data,JNU_data,MFPT_data,UoC_data,DC_data
from model.ChebyNet import ChebyNet
from model.GAT import GAT
from model.GIN import GIN
from model.GCN import GCN
from model.GraphSAGE import GraphSAGE
from model.MLP import MLP
from model.SGCN import SGCN
from model.CNN import CNN_1D
from model.LeNet import LeNet
from utils.visualization_confusion import visualization_confusion
from utils.visualization_tsne import visualization_tsne


def train_utils(args):

    #==============================================================1、训练集、测试集===================================================
    if args.dataset_name == 'CWRU':
        loader_train, loader_test = CWRU_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,dir_path=args.dir_path,
                             window_size=args.sample_length,overlap=args.overlap,normalization=args.norm_type,noise=args.noise,snr=args.snr,input_type=args.input_type,
                             graph_type=args.graph_type,K=args.knn_K,p=args.ER_p,node_num=args.node_num,direction=args.direction,
                             edge_type=args.edge_type,edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 10 #10分类

    elif args.dataset_name == 'SEU':
        loader_train, loader_test = SEU_data.data_preprocessing(dataset_path=args.dataset_path,channel=args.SEU_channel,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,
                                                                   normalization=args.norm_type, noise=args.noise,snr=args.snr,input_type=args.input_type,graph_type=args.graph_type, K=args.knn_K,
                                                                   p=args.ER_p,node_num=args.node_num, direction=args.direction,edge_type=args.edge_type,
                                                                   edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 10  # 10分类

    elif args.dataset_name == 'XJTU':
        loader_train, loader_test = XJTU_data.data_preprocessing(dataset_path=args.dataset_path,channel=args.XJTU_channel,minute_value=args.minute_value, sample_number=args.sample_num,
                                                                window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type, noise=args.noise,snr=args.snr, input_type=args.input_type,
                                                                graph_type=args.graph_type, K=args.knn_K,p=args.ER_p,node_num=args.node_num, direction=args.direction,
                                                                edge_type=args.edge_type,edge_norm=args.edge_norm, batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 15  # 15分类

    elif args.dataset_name == 'JNU':
        loader_train, loader_test = JNU_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length,overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise,snr=args.snr, input_type=args.input_type,graph_type=args.graph_type, K=args.knn_K,p=args.ER_p,
                                                                node_num=args.node_num,direction=args.direction,edge_type=args.edge_type,edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 12  # 12分类

    elif args.dataset_name == 'MFPT':
        loader_train, loader_test = MFPT_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                                node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 15  # 15分类

    elif args.dataset_name == 'UoC':
        loader_train, loader_test = UoC_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                                node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 9  # 9分类

    elif args.dataset_name == 'DC':
        loader_train, loader_test = DC_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                               noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                               node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 10  # 10分类

    else:
        print('this dataset is not existed!!!')

    #==============================================================2、网络模型===================================================
    input_dim = loader_train.dataset[0].x.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'GCN':
        model = GCN(input_dim, output_dim).to(device)
    elif args.model_type == 'ChebyNet':
        model = ChebyNet(input_dim, output_dim).to(device)
    elif args.model_type == 'GAT':
        model = GAT(input_dim, output_dim).to(device)
    elif args.model_type == 'GIN':
        model = GIN(input_dim, output_dim).to(device)
    elif args.model_type == 'GraphSAGE':
        model = GraphSAGE(input_dim, output_dim).to(device)
    elif args.model_type == 'MLP':
        model = MLP(input_dim, output_dim).to(device)
    elif args.model_type == 'SGCN':
        model = SGCN(input_dim, output_dim).to(device)
    elif args.model_type == 'LeNet':
        model = LeNet(in_channel=1,out_channel=output_dim).to(device)
    elif args.model_type == '1D-CNN':
        model = CNN_1D(in_channel=1,out_channel=output_dim).to(device)
    else:
        print('this model is not existed!!!')

    # ==============================================================3、超参数===================================================
    epochs = args.epochs
    lr = args.learning_rate
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=args.momentum)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    else:
        print('this optimizer is not existed!!!')

    # ==============================================================4、训练===================================================
    all_train_loss = []
    all_train_accuracy = []
    train_time = []

    for epoch in range(epochs):

        start = time.perf_counter()

        model.train()
        correct_train = 0
        train_loss = 0
        for step, train_data in enumerate(loader_train):
            train_data = train_data.to(device)
            train_out = model(train_data)
            loss = F.nll_loss(train_out, train_data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            pre_train = torch.max(train_out.cpu(), dim=1)[1].data.numpy()
            correct_train = correct_train + (pre_train == train_data.y.cpu().data.numpy()).astype(int).sum()

        end = time.perf_counter()

        train_time.append(end-start)  #记录训练时间

        train_accuracy = correct_train / (len(loader_train.dataset) * loader_train.dataset[0].num_nodes)
        all_train_loss.append(train_loss)
        all_train_accuracy.append(train_accuracy)

        print('epoch：{} '
              '| train loss：{:.4f} '
              '| train accuracy：{}/{}({:.4f}) '
              '| train time：{}(s/epoch)'.format(
            epoch,train_loss,correct_train,len(loader_train.dataset) * loader_train.dataset[0].num_nodes,100*train_accuracy,end-start))

    # ==============================================================5、测试===================================================
    y_fea = []
    list(map(lambda x:y_fea.append([]),range(len(model.get_fea()))))  # y_fea = [] 根据可视化的层数来创建相应数量的空列表存放特征

    prediction = np.empty(0,)  #存放预测标签绘制混淆矩阵
    model.eval()
    correct_test = 0
    for test_data in loader_test:
        test_data = test_data.to(device)
        test_out = model(test_data)
        pre_test = torch.max(test_out.cpu(),dim=1)[1].data.numpy()
        correct_test = correct_test + (pre_test == test_data.y.cpu().data.numpy()).astype(int).sum()
        prediction = np.append(prediction,pre_test) #保存预测结果---混淆矩阵
        list(map(lambda j: y_fea[j].extend(model.get_fea()[j].cpu().detach().numpy()),range(len(y_fea)))) #保存每一层特征---tsne

    test_accuracy = correct_test / (len(loader_test.dataset)*loader_test.dataset[0].num_nodes)

    print('test accuracy：{}/{}({:.4f}%)'.format(correct_test,len(loader_test.dataset*loader_test.dataset[0].num_nodes),100*test_accuracy))
    print('all train time：{}(s/100epoch)'.format(np.array(train_time).sum()))

    if args.visualization == True:

        visualization_confusion(loader_test=loader_test,prediction=prediction)  #混淆矩阵

        for num,fea in enumerate(y_fea):
            visualization_tsne(loader_test=loader_test,y_feature=np.array(fea),classes=output_dim)  #t-SNE可视化

        plt.show()


