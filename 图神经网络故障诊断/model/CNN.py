import torch.nn as nn
import torch.nn.functional as F

class CNN_1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN_1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=10,padding=1),  # 16, 26 ,26
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=6),  # 64,10,10
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4,padding=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,padding=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 128, 4,4


        self.layer6 = nn.Sequential(
            nn.Linear(64 * 30, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(256, out_channel)

    def forward(self, data):
        x = data.x.unsqueeze(dim=1)  # shape: （样本数，1，样本长度）

        self.f1 = x.reshape(x.shape[0],x.shape[1] * x.shape[2])  # 原始特征(时域输入时为时域特征，频域输入时为频域特征) --- 将通道维度展平

        x = self.layer1(x)

        self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) #第一层卷积特征

        x = self.layer2(x)

        self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # 第二层卷积特征

        x = self.layer3(x)

        self.f4 = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # 第三层卷积特征

        x = self.layer4(x)

        self.f5 = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # 第四层卷积特征

        x = self.layer5(x)

        self.f6 = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # 第五层卷积特征

        x = x.view(x.size(0), -1)

        x = self.layer6(x)
        x = self.fc(x)

        self.f7 = x.reshape(x.shape[0], x.shape[1])  # 全连接层特征

        out = F.log_softmax(x,dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4,self.f5,self.f6,self.f7]


