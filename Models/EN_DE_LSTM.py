import torch.nn as nn
from Models.model_main import Model,nnReshape,OUT_STEPS
'''编码器将输入序列的全局信息压缩到隐藏状态，解码器利用该状态初始化预测'''
class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size=64):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True,num_layers=3)

    def forward(self, x):
        # x: [batch, input_width, in_features]
        output, (hidden, cell) = self.lstm(x)
        # output: [batch, input_width, hidden_size]
        # hidden: [1, batch, hidden_size]
        # cell: [1, batch, hidden_size]
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size=64, out_features=1, out_steps=OUT_STEPS):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,num_layers=3)

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, out_steps * out_features),
            nnReshape(-1, out_steps, out_features)
        )
        nn.init.xavier_uniform_(self.linear[0].weight)

    def forward(self, input, hidden, cell):
        # input: [batch, 获取最后一个时间步的输出1, hidden_size]
        # hidden: [1, batch, hidden_size]
        # cell: [1, batch, hidden_size]
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # output: [batch, 1, hidden_size]
        prediction = self.linear(output)
        # prediction: [batch, out_steps, out_features]
        return prediction, hidden, cell


class EN_DE_LSTM(Model):
    def __init__(self, in_features, out_features, out_steps=OUT_STEPS):
        super(EN_DE_LSTM, self).__init__()
        self.encoder = Encoder(in_features)
        self.decoder = Decoder(out_features=out_features, out_steps=out_steps)

    def forward(self, x):

        # x: [batch, input_width, in_features]
        output, hidden, cell = self.encoder(x)
        # output: [batch, sequence_length, hidden_size]
        # hidden: [1, batch, hidden_size]
        # cell: [1, batch, hidden_size]

        # 获取编码器最后一个时间步的输出作为解码器的初始输入
        decoder_input = output[:, -1:, :]  # [batch, 1, hidden_size]

        # 一次性预测多步
        predictions, _, _ = self.decoder(decoder_input, hidden, cell)
        # predictions: [batch, OUT_STEPS, out_features]
        return predictions






