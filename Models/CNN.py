import torch.nn as nn
from Models.model_main import Model,nnReshape,OUT_STEPS
class MultiConv1d(Model):
    def __init__(self,in_features, num_filters, kernel_size,padding):
        super(MultiConv1d,self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters//2,
            kernel_size=kernel_size,
            padding=padding
        )
    def forward(self,x):
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_features, sequence_length)
        x = self.conv1(x)  # 第一层卷积
        x = self.relu(x)  # 激活函数
        x = self.conv2(x)  # 第二层卷积
        x = self.relu(x)  # 激活函数
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, sequence_length, num_filters//2)
        return x

class CNN(Model):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv=MultiConv1d(self.in_features,128,3,1)
        self.linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32,OUT_STEPS * out_features),
            nnReshape(-1, OUT_STEPS, out_features)
        )
    def forward(self,x):
        conv_output = self.conv(x)  # 卷积输出
        output = self.linear(conv_output[:,-1,:])       #只取最后一步
        return output