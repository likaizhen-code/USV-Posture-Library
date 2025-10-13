import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_main import Model



'''8.3，把一个特征扩展成100维，损失的太多
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
        log_probs = F.log_softmax(logits, dim=-1)  # 重要：用于 KLDivLoss
        return log_probs
'''

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