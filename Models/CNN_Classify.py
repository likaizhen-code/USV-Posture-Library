import torch.nn as nn
import torch.nn.functional as F
from Models.model_main import Model
class CNN_Classify(Model):
    def __init__(self, in_channels=100, num_bins=100, hidden_dim=64,pred_len=20,):
        super(CNN_Classify, self).__init__()

        self.num_bins = num_bins
        self.pred_len = pred_len

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(hidden_dim, 32, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合

        # 输出为 pred_len × num_bins 的 logits
        self.fc = nn.Linear(32,  num_bins)

    def forward(self, x,return_softmax=False):
        # x: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)

        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))

        x = self.dropout(x)  # 卷积后加 Dropout

        x = x.permute(0, 2, 1)

        x = self.fc(x)  # (B, pred_len * num_bins)

        if return_softmax:
            return F.softmax(x,dim=-1)
        else:
            return F.log_softmax(x,dim=-1)

'''输入格式为[Batch，Seq，高斯编码过的]'''