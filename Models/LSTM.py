
import torch.nn as nn
from Models.model_main import Model,nnReshape,OUT_STEPS


class LSTM(Model):        #单发预测
    def __init__(self, in_features, out_features):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, 64, batch_first=True, dropout=0.3, num_layers=3)
        self.linear = nn.Sequential(
            # 这里的输入数据为（batch，last1，64）
            nn.Linear(64, OUT_STEPS * out_features),
            nn.ReLU(),
            nnReshape(-1, OUT_STEPS, out_features)
        )
        nn.init.xavier_uniform_(self.linear[0].weight)

    def forward(self, x):
        # Shape [batch, seq, features] => [batch, seq, 1]
        output, _ = self.lstm(x)
        return self.linear(output[:, -1:, :])           #lstm也有hidden，输出的是一组向量需要解码
