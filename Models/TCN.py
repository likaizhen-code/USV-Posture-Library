import torch
import torch.nn as nn
from Models.model_main import Model

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] != 0 else out
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] != 0 else out
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return out


class TCN(Model):
    def __init__(self, input_size, num_channels, kernel_size, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 20)

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)  # 转为 [batch_size, input_size, seq_len]
        out = self.network(x)
        # 取最后一个时间步的特征
        out = out[:, :, -1]
        # 通过全连接层得到一个输出
        out = self.fc(out)
        return out.unsqueeze(-1)