import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_main import Model


class CNN_Classify(nn.Module):
    def __init__(self, in_channels=100, hidden_dim=64,out_channels=32,pred_len=10):
        super(CNN_Classify, self).__init__()

        self.pred_len = pred_len

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)

        # self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合


    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)

        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))

        # x = self.dropout(x)  # 卷积后加 Dropout

        x = x.permute(0, 2, 1)
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



class CNN_LSTM_Classify(Model):
    def __init__(self,pred_len,num_classes=100):
        super(CNN_LSTM_Classify, self).__init__()
        self.lstm_classify = LSTM_Classify(pred_len=pred_len)
        self.cnn_classify = CNN_Classify(pred_len=pred_len)
        self.fc = nn.Linear(42,  num_classes)    # ...+...改完输入输出数后接着改这里


    def forward(self, x,return_softmax=False):
        x_cnn = x[:,:,1:]
        x_cnn = self.cnn_classify(x_cnn)

        x_lstm = x[:,:,0].unsqueeze(-1)
        x_lstm = self.lstm_classify(x_lstm)

        x=torch.cat((x_lstm,x_cnn),dim=-1)
        x=self.fc(x)

        if return_softmax:
            return F.softmax(x, dim=-1)
        else:
            return F.log_softmax(x, dim=-1)

