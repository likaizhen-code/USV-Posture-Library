import torch
from torch import nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_STEPS=10

class nnReshape(nn.Module):       #将输入张量重新组织为形状为 [batch, OUT_STEPS, out_features] 的张量
    def __init__(self, *shape):
        super(nnReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)
# class Model(nn.Module):
#     def compile(self,loss_fn,metric_fn,optimizer=None,scheduler=None):    #metric评估指标，optimizer优化器
#         self.loss_fn = loss_fn
#         self.metric_fn = metric_fn
#         self.optimizer =optimizer
#         self.scheduler = scheduler
#
#     def train_epoch(self, dataloader):
#         self.train()
#         avg_loss = 0
#         avg_metric = 0
#
#         for x, y in dataloader:
#             x=x.to(device)
#             y = y.to(device)
#
#             yp = self(x)                        #窗口截取一部分，x是输入，y是输出，yp是经过模型的输出
#             yp = yp[:, :, 0].unsqueeze(-1)
#             y=y[:,:,0].unsqueeze(-1)         #选取了一列之后，就会变成二维
#             loss = self.loss_fn(y, yp)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             avg_loss += loss.item()
#             avg_metric += self.metric_fn(y, yp).item()
#
#         num_batches = len(dataloader)
#         avg_loss /= num_batches
#         avg_metric /= num_batches
#
#         return avg_loss, avg_metric
#
#     @torch.no_grad()
#     def evaluate(self, dataloader):
#         self.eval()
#         avg_loss = 0
#         avg_metric = 0
#
#         for x, y in dataloader:
#             x=x.to(device)
#             y = y.to(device)
#
#             yp = self(x)
#             yp = yp[:, :, 0].unsqueeze(-1)
#
#             y = y[:, :, 0].unsqueeze(-1)  # 选取了一列之后，就会变成二维
#             avg_loss += self.loss_fn(y, yp).item()
#             avg_metric += self.metric_fn(y, yp).item()
#
#         num_batchs = len(dataloader)
#         avg_loss /= num_batchs
#         avg_metric /= num_batchs
#
#         return avg_loss, avg_metric



def dynamic_smooth_encoding_torch(values,
                                  min_val=-15,
                                  max_val=15,
                                  num_bins=30,
                                  alpha=0.3):
    """
    对三维张量 (B, L, 1) 或 (B, L) 进行编码。
    保留主 bin 的权重，同时根据值在 bin 内的位置选择一个邻区间（左或右）
    并给该邻区间一个比例（由 alpha 调节，取值 0..1）。
    输出形状: (B, L, num_bins)
    """
    if values.dim() == 3 and values.size(-1) == 1:
        values = values.squeeze(-1)

    B, L = values.shape
    values = torch.clamp(values, min=min_val, max=max_val - 1e-8)
    bin_width = (max_val - min_val) / num_bins
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=values.device)

    # 当前 bin 索引（0..num_bins-1）
    bin_idx = torch.bucketize(values, bin_edges, right=False) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    # 在当前 bin 内的相对位置 ratio ∈ [0,1)
    dist_to_left = values - bin_edges[bin_idx]
    ratio = (dist_to_left / bin_width).clamp(0.0, 1.0 - 1e-8)

    # 计算邻区强度（只选一侧）
    # 当 ratio < 0.5 -> 左邻强度 s = 1 - 2*ratio
    # 当 ratio >= 0.5 -> 右邻强度 s = 2*ratio - 1
    left_strength = (1.0 - 2.0 * ratio).clamp(min=0.0)   # 在右半区为0
    right_strength = (2.0 * ratio - 1.0).clamp(min=0.0)  # 在左半区为0

    # 邻区实际权重 = alpha * strength；主区权重 = 1 - neighbor_weight
    neighbor_weight_left = alpha * left_strength
    neighbor_weight_right = alpha * right_strength
    neighbor_weight = neighbor_weight_left + neighbor_weight_right
    main_weight = (1.0 - neighbor_weight).clamp(min=0.0)

    # 初始化编码
    encoded = torch.zeros(B, L, num_bins, device=values.device, dtype=values.dtype)

    # 加到主 bin
    encoded.scatter_add_(2, bin_idx.unsqueeze(-1), main_weight.unsqueeze(-1))

    # 加到左邻（仅当存在左邻时）
    left_mask = bin_idx > 0
    left_idx = (bin_idx - 1).clamp(0, num_bins - 1)
    left_add = torch.where(left_mask, neighbor_weight_left, torch.zeros_like(neighbor_weight_left))
    encoded.scatter_add_(2, left_idx.unsqueeze(-1), left_add.unsqueeze(-1))

    # 加到右邻（仅当存在右邻时）
    right_mask = bin_idx < (num_bins - 1)
    right_idx = (bin_idx + 1).clamp(0, num_bins - 1)
    right_add = torch.where(right_mask, neighbor_weight_right, torch.zeros_like(neighbor_weight_right))
    encoded.scatter_add_(2, right_idx.unsqueeze(-1), right_add.unsqueeze(-1))

    # 若边界导致总和 < 1（例如没有可用邻区），可选择让主区补齐或归一化
    # 这里统一做归一化以防数值误差
    encoded = encoded / encoded.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return encoded


#回归+分类11.11
class Model(nn.Module):
    def compile(self,value_loss,conf_loss,align_loss,metric_fn,optimizer=None,scheduler=None):    #metric评估指标，optimizer优化器
        self.value_loss = value_loss
        self.conf_loss = conf_loss
        self.align_loss = align_loss

        self.metric_fn = metric_fn
        self.optimizer =optimizer
        self.scheduler = scheduler

    def train_epoch(self, dataloader):
        print('回归+分类训练任务')
        self.train()
        self.metric_fn.reset()  # 重置指标
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            self.optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean
            y = y - x_mean
            yp = self(x)

            yp_regress = yp[:, :, 0].unsqueeze(-1)
            yp_label = yp[:, :, 1:]

            offset = yp_regress + 15  # 距离左端点-15的偏移量
            index_float = offset / 0.75  # 浮点数索引
            index_regress = torch.clamp(index_float, 0, 30).float()
            _,index_label = torch.topk(yp_label, k=1, dim=2)
            index_label =index_label.float()

            y_regress = y
            y_label  = dynamic_smooth_encoding_torch(y,num_bins=40)       # 编码



            loss = self.value_loss(yp_regress, y_regress) + self.conf_loss(yp_label, y_label) * 0.5 + self.align_loss(index_regress, index_label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.metric_fn.to(device).update(yp_regress, y)

        avg_loss = total_loss / num_batches
        metric = self.metric_fn.compute()
        return avg_loss,metric

    def evaluate(self,dataloader):
        self.eval()
        self.metric_fn.reset()  # 重置指标
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                self.optimizer.zero_grad()

                x, y = x.to(device), y.to(device)

                x_mean = x.mean(dim=1, keepdim=True)
                x = x - x_mean
                y = y - x_mean
                yp = self(x)

                yp_regress = yp[:, :, 0].unsqueeze(-1)
                yp_label = yp[:, :, 1:]

                y_regress = y
                y_label = dynamic_smooth_encoding_torch(y, num_bins=40)  # 编码

                loss = self.value_loss(yp_regress, y_regress) + self.conf_loss(yp_label, y_label) * 0.5

                total_loss += loss.item()
                num_batches += 1
                self.metric_fn.to(device).update(yp_regress, y)

        avg_loss = total_loss / num_batches
        metric = self.metric_fn.compute()
        return avg_loss, metric




#分类任务11.1
# class Model(nn.Module):
#     def compile(self,loss_fn,metric_fn,optimizer=None,scheduler=None):    #metric评估指标，optimizer优化器
#         self.loss_fn = loss_fn
#         self.metric_fn = metric_fn
#         self.optimizer =optimizer
#         self.scheduler = scheduler
#
#     def train_epoch(self, dataloader):
#         print('分类训练任务')
#         self.train()
#         self.metric_fn.reset()  # 重置指标
#         total_loss = 0.0
#         num_batches = 0
#
#         for x, y in dataloader:
#             self.optimizer.zero_grad()
#
#             x, y = x.to(device), y.to(device)
#
#             x_mean = x.mean(dim=1, keepdim=True)
#             x = x - x_mean
#             y = y - x_mean
#             yp = self(x)
#
#             y  = dynamic_smooth_encoding_torch(y,num_bins=30)       # 编码
#
#             loss = self.loss_fn(yp, y)
#             loss.backward()
#             self.optimizer.step()
#
#             total_loss += loss.item()
#             num_batches += 1
#             self.metric_fn.to(device).update(yp, y)
#
#         avg_loss = total_loss / num_batches
#         metric = self.metric_fn.compute()
#         return avg_loss,metric
#
#     def evaluate(self,dataloader):
#         self.eval()
#         self.metric_fn.reset()  # 重置指标
#         total_loss = 0.0
#         num_batches = 0
#
#         with torch.no_grad():
#             for x, y in dataloader:
#
#                 x, y = x.to(device), y.to(device)
#
#                 x_mean = x.mean(dim=1, keepdim=True)
#                 x = x - x_mean
#                 y = y - x_mean
#                 yp = self(x)
#
#                 y = dynamic_smooth_encoding_torch(y, num_bins=30)  # 编码
#                 loss = self.loss_fn(yp, y)
#
#                 total_loss += loss.item()
#                 num_batches += 1
#                 self.metric_fn.to(device).update(yp, y)
#
#         avg_loss = total_loss / num_batches
#         metric = self.metric_fn.compute()
#         return avg_loss, metric

#分类任务8.7
# class Model(nn.Module):
#     def compile(self,loss_fn,metric_fn,optimizer=None,scheduler=None):    #metric评估指标，optimizer优化器
#         self.loss_fn = loss_fn
#         self.metric_fn = metric_fn
#         self.optimizer =optimizer
#         self.scheduler = scheduler
#
#     def train_epoch(self, dataloader):
#         self.train()
#         self.metric_fn.reset()  # 重置指标
#         total_loss = 0.0
#         num_batches = 0
#
#         for x, y in dataloader:
#             self.optimizer.zero_grad()
#
#             x, y = x.to(device), y.to(device)
#             yp = self(x)
#
#
#             y=y[:,:,1:]
#             # target_probs 必须是概率分布且避免 0
#             eps = 1e-8
#             y = y.clamp(min=eps)
#             y = y / y.sum(dim=-1, keepdim=True)
#
#             loss = self.loss_fn(yp, y)
#             loss.backward()
#             self.optimizer.step()
#
#             total_loss += loss.item()
#             num_batches += 1
#             self.metric_fn.to(device).update(yp, y)
#
#         avg_loss = total_loss / num_batches
#         metric = self.metric_fn.compute()
#         return avg_loss,metric
#
#     def evaluate(self,dataloader):
#         self.eval()
#         self.metric_fn.reset()  # 重置指标
#         total_loss = 0.0
#         num_batches = 0
#
#         with torch.no_grad():
#             for x, y in dataloader:
#
#                 x, y = x.to(device), y.to(device)
#                 yp = self(x)
#
#                 y = y[:, :, 1:]
#                 # target_probs 必须是概率分布且避免 0
#                 eps = 1e-8
#                 y = y.clamp(min=eps)
#                 y = y / y.sum(dim=-1, keepdim=True)
#
#                 loss = self.loss_fn(yp, y)
#
#                 total_loss += loss.item()
#                 num_batches += 1
#                 self.metric_fn.to(device).update(yp, y)
#
#         avg_loss = total_loss / num_batches
#         metric = self.metric_fn.compute()
#         return avg_loss, metric


# 中心化
# class Model(nn.Module):
#     def compile(self,loss_fn,metric_fn,optimizer=None,scheduler=None):    #metric评估指标，optimizer优化器
#         self.loss_fn = loss_fn
#         self.metric_fn = metric_fn
#         self.optimizer =optimizer
#         self.scheduler = scheduler
#
#     def train_epoch(self, dataloader):
#         self.train()
#
#         avg_loss = 0
#         avg_metric = 0
#
#         for x, y in dataloader:
#             x=x.to(device)
#             y = y.to(device)
#
#             x_mean = x.mean(dim=1, keepdim=True)
#             x_std = 1
#
#             x = x - x_mean
#             y = y - x_mean
#             x = x / x_std
#             y = y /x_std
#
#             yp = self(x)                             #窗口截取一部分，x是输入，y是输出，yp是经过模型的输出
#             y=y[:,:,0].unsqueeze(-1)         #选取了一列之后，就会变成二维
#             loss = self.loss_fn(y, yp)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             avg_loss += loss.item()
#             avg_metric += self.metric_fn(y, yp).item()
#
#         num_batches = len(dataloader)
#         avg_loss /= num_batches
#         avg_metric /= num_batches
#
#         return avg_loss, avg_metric
#
#
#     @torch.no_grad()
#     def evaluate(self, dataloader):
#         self.eval()
#         avg_loss = 0
#         avg_metric = 0
#
#         for x, y in dataloader:
#             x=x.to(device)
#             y = y.to(device)
#
#             x_mean = x.mean(dim=1, keepdim=True)
#             x_std = 1
#
#             x = x - x_mean
#             y = y - x_mean
#             x = x / x_std
#             y = y /x_std
#
#             yp = self(x)
#             y = y[:, :, 0].unsqueeze(-1)  # 选取了一列之后，就会变成二维
#             avg_loss += self.loss_fn(y, yp).item()
#             avg_metric += self.metric_fn(y, yp).item()
#
#         num_batchs = len(dataloader)
#         avg_loss /= num_batchs
#         avg_metric /= num_batchs
#
#         return avg_loss, avg_metric
