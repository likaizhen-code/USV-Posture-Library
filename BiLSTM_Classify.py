import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_main import Model,nnReshape

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x    # 获得移动平均值


class BiLSTM_Classify(Model):
    def __init__(self, in_features=1, input_len=20, pred_len=10, hidden_size=64, num_classes=30, temperature=0.5):
        super().__init__()
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.temperature = temperature

        # 1. 核心时序引擎 (Encoder-Decoder)
        self.encoder = nn.LSTM(in_features, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size, num_layers=3, batch_first=True, bidirectional=True)

        # 2. 预测头：合并输出以简化 forward 逻辑
        # 总输出维度 = value_dim(1) + num_classes
        bi_hidden = hidden_size * 2
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bi_hidden, 1 + num_classes)
        )

        # 3. 趋势分支
        self.decompsition = moving_avg(kernel_size=3, stride=1)
        self.trend_linear = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 预测均值偏移量
        )

    def forward(self, x, return_softmax=False):
        B, T, C = x.shape

        # --- A. 均值与趋势处理 ---
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, C]

        # 趋势偏移预测: (B, C, T) -> (B, C, 1) -> (B, 1, C)
        trend = self.decompsition(x).permute(0, 2, 1)
        delta_mean = self.trend_linear(trend - x_mean).transpose(1, 2)
        pred_mean = x_mean + delta_mean

        # --- B. 波动项建模 (LSTM) ---
        # 编码残差
        enc_out, (h, c) = self.encoder(x - x_mean)

        # 解码预测未来 pred_len 个步长
        # 构造 decoder 输入：取 encoder 最后一个 step 并重复 pred_len 次（或使用 autoregressive）
        # 这里采用一次性展开方式以匹配您原有的 predictions 逻辑
        dec_in = enc_out[:, -1:, :].expand(-1, self.pred_len, -1)
        dec_out, _ = self.decoder(dec_in, (h, c))

        # --- C. 多任务输出 ---
        # 结果维度: [B, pred_len, 1 + num_classes]
        out = self.head(dec_out)

        # 拆分 Value (波动 + 均值) 与 Confidence (Logits)
        value = out[..., :1] + pred_mean
        conf_logits = out[..., 1:] / self.temperature

        # --- D. 返回处理 ---
        conf = F.softmax(conf_logits, dim=-1) if return_softmax else F.log_softmax(conf_logits, dim=-1)

        return torch.cat([value, conf], dim=-1)