import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(GIN, self).__init__()

        self.GConv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)))

        self.bn1 = BatchNorm(1024)

        self.GConv2 = GINConv(
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)))

        self.bn2 = BatchNorm(1024)

        self.linear = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )


    def forward(self, data):
        x, edge_index= data.x, data.edge_index

        self.f1 = x  # 原始特征(时域输入时为时域特征，频域输入时为频域特征)

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        self.f2 = x  # 第二层特征

        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        self.f3 = x  # 第二层特征

        x = self.linear(x)

        self.f4 = x  # 全连接层特征

        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]