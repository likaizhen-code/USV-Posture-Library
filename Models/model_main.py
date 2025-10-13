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


#分类任务8.7
class Model(nn.Module):
    def compile(self,loss_fn,metric_fn,optimizer=None,scheduler=None):    #metric评估指标，optimizer优化器
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer =optimizer
        self.scheduler = scheduler

    def train_epoch(self, dataloader):
        self.train()
        self.metric_fn.reset()  # 重置指标
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            self.optimizer.zero_grad()

            x, y = x.to(device), y.to(device)
            yp = self(x)


            y=y[:,:,1:]
            # target_probs 必须是概率分布且避免 0
            eps = 1e-8
            y = y.clamp(min=eps)
            y = y / y.sum(dim=-1, keepdim=True)

            loss = self.loss_fn(yp, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.metric_fn.to(device).update(yp, y)

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

                x, y = x.to(device), y.to(device)
                yp = self(x)

                y = y[:, :, 1:]
                # target_probs 必须是概率分布且避免 0
                eps = 1e-8
                y = y.clamp(min=eps)
                y = y / y.sum(dim=-1, keepdim=True)

                loss = self.loss_fn(yp, y)

                total_loss += loss.item()
                num_batches += 1
                self.metric_fn.to(device).update(yp, y)

        avg_loss = total_loss / num_batches
        metric = self.metric_fn.compute()
        return avg_loss, metric



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
