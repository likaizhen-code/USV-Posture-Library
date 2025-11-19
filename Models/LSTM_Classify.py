import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_main import Model


'''多输出lstm结构（回归+分类）'''
class LSTM_Detect(Model):
    def __init__(self, in_features=1, pred_len=10, value_dim=1, hidden_size=64, num_classes=30, temperature=0.5):
        super(LSTM_Detect, self).__init__()
        self.pred_len = pred_len
        self.value_dim = value_dim
        self.num_classes = num_classes
        self.temperature = temperature  # ⭐ 固定温度系数

        # LSTM
        self.lstm_encoder = nn.LSTM(in_features, hidden_size, batch_first=True, num_layers=3)
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=3)

        # 输出头
        self.value_head = nn.Linear(hidden_size, self.pred_len * self.value_dim)
        self.conf_head  = nn.Linear(hidden_size, self.pred_len * self.num_classes)

    def forward(self, x, return_softmax=False):

        x = x[:, :, 0].unsqueeze(-1)

        x, (hidden, cell) = self.lstm_encoder(x)
        x = x[:, -1:, :]
        x, (hidden, cell) = self.lstm_decoder(x, (hidden, cell))

        # 回归头
        value = self.value_head(x).view(x.size(0), self.pred_len, self.value_dim)

        # 分类 logits
        conf_logits = self.conf_head(x).view(x.size(0), self.pred_len, self.num_classes)

        # ⭐ 固定温度缩放（直接除）
        conf_logits = conf_logits / self.temperature

        # softmax 或 log_softmax
        if return_softmax:
            conf = F.softmax(conf_logits, dim=-1)
        else:
            conf = F.log_softmax(conf_logits, dim=-1)

        return torch.cat([value, conf], dim=-1)
