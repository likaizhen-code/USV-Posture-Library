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




'''phi输入'''
# class LSTM_Classify(Model):
#     def __init__(self, in_features=1, pred_len=10,num_classes=30, hidden_size=64):
#         super(LSTM_Classify, self).__init__()
#         self.pred_len = pred_len
#         self.num_classes = num_classes
#         self.lstm_encoder = nn.LSTM(in_features, hidden_size, batch_first=True, num_layers=3)
#         self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=3)
#         self.linear = nn.Linear(hidden_size, self.pred_len*self.num_classes)
#
#     def forward(self, x,return_softmax=False):
#         x= x[:,:,0].unsqueeze(-1)   #限制只有横摇
#         x,(hidden, cell) = self.lstm_encoder(x)
#         x =x[:,-1:,:]
#         x, (hidden, cell) = self.lstm_decoder(x, (hidden, cell))
#         x = self.linear(x)
#         x = x.view(x.size(0),self.pred_len,-1)  # 10步为10，20步为5
#         if return_softmax:
#             return F.softmax(x, dim=-1)
#         else:
#             return F.log_softmax(x, dim=-1)


'''#7.29
class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size=64):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True, num_layers=3)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell
class LSTM_Classify(Model):
    def __init__(self, in_features, num_classes=100, hidden_size=64):
        super(LSTM_Classify, self).__init__()
        self.encoder = Encoder(in_features, hidden_size)
        self.step_classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _, _ = self.encoder(x)  # [batch, seq_len, hidden_size]
        logits = self.step_classifier(output)  # [batch, seq_len, num_classes]
        return logits  # ⚠ 注意这里返回 raw logits，不加 softmax
'''

'''#7.27
class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size=64):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True, num_layers=3)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell

class LSTM_Classify(Model):
    def __init__(self, in_features, num_classes=100, hidden_size=64):
        super(LSTM_Classify, self).__init__()
        self.encoder = Encoder(in_features, hidden_size)

        self.step_classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, in_features]
        输出:
            probabilities: [batch_size, seq_len, num_classes]
        """
        output, _, _ = self.encoder(x)  # [batch, seq_len, hidden_size]
        logits = self.step_classifier(output)  # [batch, seq_len, num_classes]

        # ⚠ 关键：softmax 生成概率分布用于 MSELoss
        probs = F.softmax(logits, dim=-1)

        return probs
'''