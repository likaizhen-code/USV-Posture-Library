import torch
import torch.nn.functional as F
from torch import nn
from Models.model_main import Model


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 卷积匹配维度（残差连接用）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res
class TCN_Classify(nn.Module):
    def __init__(self, in_channels=100, num_channels=[64, 32], kernel_size=3, dropout=0.2, pred_len=10):
        super(TCN_Classify, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation//2
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.pred_len = pred_len

    def forward(self, x):
        # 输入 (B, T, C) → 转换为 (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.network(x)   # (B, C_out, T)
        x = x.permute(0, 2, 1)  # (B, T, C_out)
        return x


class LSTM_Classify(nn.Module):
    def __init__(self, in_features=1, hidden_size=128,pred_len=10):
        super(LSTM_Classify, self).__init__()
        self.pred_len = pred_len

        self.lstm_encoder = nn.LSTM(in_features, hidden_size, batch_first=True, num_layers=3)
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=3)
        self.linear = nn.Linear(hidden_size, 100)
    def forward(self, x):
        x,(hidden, cell) = self.lstm_encoder(x)
        x =x[:,-1:,:]
        x, (hidden, cell) = self.lstm_decoder(x, (hidden, cell))
        x = self.linear(x)
        x = x.view(x.size(0),self.pred_len,-1)
        return x


class TCN_LSTM_Classify(Model):
    def __init__(self, pred_len, num_classes=100):
        super(TCN_LSTM_Classify, self).__init__()
        self.lstm_classify = LSTM_Classify(pred_len=pred_len)
        self.tcn_classify = TCN_Classify(in_channels=num_classes, num_channels=[64, 32], pred_len=pred_len)

        # LSTM 输出 = (B, pred_len, 10)
        # TCN 输出 = (B, pred_len, 32)
        self.fc = nn.Linear( 10 + 32, num_classes)   #改：  10步为10，20步为5

    def forward(self, x, return_softmax=False):
        x_tcn = self.tcn_classify(x[:,:,1:])   # (B, T, 32)
        x_lstm = self.lstm_classify(x[:,:,0].unsqueeze(-1))  # (B, T, 100)

        x = torch.cat((x_lstm, x_tcn), dim=-1)  # (B, T, 132)
        x = self.fc(x)

        if return_softmax:
            return F.softmax(x, dim=-1)
        else:
            return F.log_softmax(x, dim=-1)
