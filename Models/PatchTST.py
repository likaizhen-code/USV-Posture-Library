from Models.model_main import Model,nnReshape
from Models import EN_DE_LSTM
# from statsmodels.tsa.seasonal import STL


# Cell
from typing import Callable, Optional
import torch
from torch import nn, lstm
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class MOE_Config:
    def __init__(self):
        # 基本配置
        self.enc_in = num_features  # 输入通道数/变量数
        self.seq_len = INPUT_STEPS  # 输入序列长度
        self.pred_len = OUT_STEPS  # 预测序列长度

        # PatchTST 结构参数
        self.e_layers = 2  # 编码器层数
        self.n_heads = 4  # 注意力头数
        self.d_model = 32  # 模型维度
        self.d_ff = 4  # 前馈网络维度
        self.dropout = 0.2  # dropout率
        self.fc_dropout = 0.3  # 全连接层dropout率
        self.head_dropout = 0.3  # 头部dropout率

        # Patch相关参数
        self.patch_len = 5  # patch长度
        self.stride = 1  # patch步长
        self.padding_patch = 'end'  # patch填充方式

        # 其他参数
        self.individual = False  # 是否为每个通道使用独立的头部
        self.revin = True  # 是否使用可逆实例归一化
        self.affine = True  # 是否在RevIN中使用仿射变换
        self.subtract_last = True  # 是否减去最后一个值而不是平均值

        # 分解相关
        self.decomposition = True  # 是否使用序列分解
        self.kernel_size = 5  # 分解的核大小
class Config_Real:
    def __init__(self):
        # 基本配置
        self.enc_in = num_features  # 输入通道数/变量数
        self.seq_len = INPUT_STEPS  # 输入序列长度
        self.pred_len = OUT_STEPS  # 预测序列长度

        # PatchTST 结构参数
        self.e_layers = 2  # 编码器层数
        self.n_heads = 4  # 注意力头数
        self.d_model = 32  # 模型维度
        self.d_ff = 4  # 前馈网络维度
        self.dropout = 0.1  # dropout率
        self.fc_dropout = 0.2  # 全连接层dropout率
        self.head_dropout = 0.2  # 头部dropout率

        # Patch相关参数
        self.patch_len = 5  # patch长度
        self.stride = 3  # patch步长
        self.padding_patch = 'end'  # patch填充方式

        # 其他参数
        self.individual = False  # 是否为每个通道使用独立的头部
        self.revin = False  # 是否使用可逆实例归一化
        self.affine = True  # 是否在RevIN中使用仿射变换
        self.subtract_last = False  # 是否减去最后一个值而不是平均值

        # 分解相关
        self.decomposition = True  # 是否使用序列分解
        self.kernel_size = 13  # 分解的核大小

class Config:
    def __init__(self):
        # 基本配置
        self.enc_in = num_features  # 输入通道数/变量数
        self.seq_len = INPUT_STEPS  # 输入序列长度
        self.pred_len = OUT_STEPS  # 预测序列长度

        # PatchTST 结构参数
        self.e_layers = 2  # 编码器层数
        self.n_heads = 4  # 注意力头数
        self.d_model = 32  # 模型维度
        self.d_ff = 4  # 前馈网络维度
        self.dropout = 0.1  # dropout率
        self.fc_dropout = 0.2  # 全连接层dropout率
        self.head_dropout = 0.2  # 头部dropout率

        # Patch相关参数
        self.patch_len = 5  # patch长度
        self.stride = 3  # patch步长
        self.padding_patch = 'end'  # patch填充方式

        # 其他参数
        self.individual = False  # 是否为每个通道使用独立的头部
        self.revin = False  # 是否使用可逆实例归一化
        self.affine = True  # 是否在RevIN中使用仿射变换
        self.subtract_last = False  # 是否减去最后一个值而不是平均值

        # 分解相关
        self.decomposition = True  # 是否使用序列分解
        self.kernel_size = 5  # 分解的核大小

class PatchTST(Model):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                                 patch_len=patch_len, stride=stride,
                                                 max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                 n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=attn_dropout,
                                                 dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                 padding_var=padding_var,
                                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                 store_attn=store_attn,
                                                 pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                 head_dropout=head_dropout, padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head, head_type=head_type,
                                                 individual=individual, revin=revin, affine=affine,
                                                 subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                               head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                               revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.lstm = nn.LSTM(1, 64, batch_first=True, dropout=0.3, num_layers=3)
            self.linear = nn.Sequential(
                # 这里的输入数据为（batch，last1，64）
                nn.Linear(64, 20),
                nn.ReLU(),
                nnReshape(-1, 20, 1)
            )
            self.en_de_lstm =EN_DE_LSTM.EN_DE_LSTM(1,1)

        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:


            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2,
                                                                                 1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)



            # res,_= self.lstm(res_init)
            # res=self.linear(res[:,-1,:])
            # trend,_= self.lstm(trend_init)
            # trend= self.linear(trend[:,-1,:])

            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]

        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x

'''
    res_init, trend_init = self.decomp_module(x)
    res=self.en_de_lstm(res_init)
    trend=self.en_de_lstm(trend_init)
    x = res + trend
'''

'''            x=x.cpu().numpy()
            stl = STL(x, period=20)  # 周期需根据实际数据调整
            res = stl.fit()
            res.to(self.device)

            trend,_= self.lstm(res.trend)
            trend=self.linear(trend[:,-1,:])

            seasonal= self.lstm(res.seasonal)
            seasonal= self.linear(seasonal[:,-1,:])

            resid= self.lstm(res.resid)
            resid= self.linear(resid[:,-1,:])
            x=trend+ seasonal +resid
'''