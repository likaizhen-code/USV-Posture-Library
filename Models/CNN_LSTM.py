import torch
import torch.nn as nn
from Models.model_main import Model,nnReshape,OUT_STEPS

'''并行'''
class MultiLstmModel(Model):        #单发预测
    def __init__(self, in_features):
        super(MultiLstmModel, self).__init__()
        self.lstm = nn.LSTM(in_features, 64, batch_first=True)

    def forward(self, x):
        # Shape [batch, seq, features] => [batch, seq, 1]
        output, _ = self.lstm(x)
        return output[:, :, :]       #lstm也有hidden，输出的是一组向量需要解码,全部取出

class MultiConv1d(Model):
    def __init__(self,in_features, num_filters, kernel_size, padding):
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

class CNN_LSTM(Model):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm=MultiLstmModel(self.in_features)
        self.conv=MultiConv1d(self.in_features,128,3,1)
        self.linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64,OUT_STEPS * out_features),
            nnReshape(-1, OUT_STEPS, out_features)
        )
    def forward(self,x):
        lstm_output = self.lstm(x)  # LSTM 输出
        conv_output = self.conv(x)  # 卷积输出
        combined_output = torch.cat([lstm_output, conv_output], dim=-1)
        output = self.linear(combined_output[:,-1,:])       #只取最后一步
        return output



'''串行'''

class lstm(Model):        #单发预测
    def __init__(self, in_features, out_features):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(in_features, 64, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # 加入 Dropout，丢弃概率为 0.5
        self.linear = nn.Sequential(
            # 这里的输入数据为（batch，last1，64）
            nn.Linear(64, OUT_STEPS * out_features),
            nnReshape(-1, OUT_STEPS, out_features)
        )
        nn.init.xavier_uniform_(self.linear[0].weight)

    def forward(self, x):
        # Shape [batch, seq, features] => [batch, seq, 1]
        output, _ = self.lstm(x)
        return self.linear(output[:, -1:, :])           #lstm也有hidden，输出的是一组向量需要解码

class conv(Model):
    def __init__(self,in_features, num_filters, kernel_size, padding):
        super(conv,self).__init__()
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

class cnn_lstm(Model):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm=lstm(self.in_features,self.out_features)
        self.conv=conv(self.in_features,64,3,1)
        self.linear = nn.Sequential(
            nn.Linear(32, 1)
        )
    def forward(self,x):
        conv_output = self.conv(x)  # 卷积输出
        linear_output = self.linear(conv_output)
        lstm_output= self.lstm(linear_output)
        # en_de_lstm=EN_DE_LSTM.EN_DE_LSTM(1,1)
        # en_de_lstm.to(torch.device("cuda"))
        # lstm_output = en_de_lstm(linear_output)  # LSTM 输出
        return lstm_output
