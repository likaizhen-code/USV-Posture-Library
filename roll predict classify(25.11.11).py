'''真实数据,采样频率0.05s/20Hz，周期2s'''
'''用时序分类方法做预测'''
'''回归+分类+中心化'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]

df = pd.read_csv(r"E:\shiyan\ShipMotion_Real\9-Still-NYN-20221212-1051.csv")
plot_cols = ['phi']
df = df[plot_cols]
#采样频率降为0.1s
df=df[::2]



n = len(df)
i1 = int(n * 0.7)
i2 = int(n * 0.9)
train_df =df.iloc[:i1]
val_df = df.iloc[i1:i2]
test_df = df[i2:]

class TimeseriesDataset(Dataset):
    def __init__(self,data,window,transform=None):
        self.data=torch.tensor(data,dtype=torch.float)
        self.window=window
        self.transform=transform
    def __len__(self):
        return len(self.data)-self.window+1
    def __getitem__(self,index):
        if index < 0:
            index += len(self)
        features = self.data[index:index + self.window]

        if self.transform is not None:
            features = self.transform(features)

        return features


class WindowGenerator:
    def __init__(self,input_width,label_width,shift, train_df=train_df,val_df=val_df,test_df=test_df,label_columns=None):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.train_df =train_df
        self.val_df = val_df
        self.test_df = test_df
        self.columns = train_df.columns

        if label_columns is None:
            self.label_columns =self.columns
        else:
            self.label_columns =pd.Index(label_columns)  #创建索引


        self.label_columns_indices=[self.columns.get_loc(name) for name in self.label_columns]   #找到索引的那一列

        self.total_window_size=self.input_width+self.shift
        self.label_start = self.total_window_size - self.label_width

        self.input_slice=slice(input_width)#切input长度
        self.input_indices = np.arange(input_width) #默认起点为0，步长为1，输出input长度的数组

        self.label_slice=slice(self.label_start,None)
        self.label_indices = np.arange(self.label_start, self.total_window_size)

    def __repr__(self):#创建实例后打印实例
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column names(s): {self.label_columns.to_list()}'
        ])

    def split_window(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.label_slice, self.label_columns_indices]

        return inputs, labels

    def make_dataloader(self,df):
        data=df.to_numpy()
        dataset=TimeseriesDataset(
                               data=data,
                               window=self.total_window_size,
                               transform=self.split_window)
        dataloader =DataLoader(dataset=dataset,
                              batch_size=64,
                              shuffle=True) #随机排列
        return dataloader

    def make_dataloader_test(self,df):
        data=df.to_numpy()
        dataset=TimeseriesDataset(
                               data=data,
                               window=self.total_window_size,
                               transform=self.split_window)
        dataloader =DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False)
        return dataloader
    @property
    def train(self):
        return self.make_dataloader(self.train_df)

    @property
    def val(self):
        return self.make_dataloader(self.val_df)

    @property
    def test(self):
        return self.make_dataloader_test(self.test_df)  #只用来测试指标，不画图


    def plot(self, model=None):        #不重叠地画图
        # 保证 test_df 行数为OUT_STEPS的倍数，多余部分截掉
        total_len = len(self.test_df)
        new_len = total_len - (total_len % OUT_STEPS)
        self.test_df = self.test_df.iloc[:new_len]

        inputs=self.test_df[:-OUT_STEPS]
        inputs = torch.tensor(inputs.values, dtype=torch.float).to(device)
        N,F = inputs.shape
        B=N//OUT_STEPS
        inputs = inputs.reshape(B,OUT_STEPS,F)
        inputs=inputs[:,:,:]


        labels=self.test_df[OUT_STEPS:]
        labels = torch.tensor(labels.values, dtype=torch.float).to(device)
        N,F = labels.shape
        B=N//OUT_STEPS
        labels = labels.reshape(B,OUT_STEPS,F)
        labels=labels[:,:,0].unsqueeze(-1)


        if model is not None:
            model.eval()  # 防止未训练时想绘图
            model.to(device)
            predictions_list=[]
            confidence_list=[]
            batch_size = inputs.shape[0]

            with torch.no_grad():
                for i in range(batch_size):
                    batch = inputs[i].unsqueeze(0)  # 取出第i个batch，增加batch维度

                    #修改输入数据格式
                    batch_mean = batch.mean(dim=1, keepdim=True)
                    batch = batch - batch_mean  #中心化
                    batch_pred = model(batch, return_softmax=True)

                    batch_regress = batch_pred[:, :, 0].unsqueeze(-1)
                    batch_conf = batch_pred[:, :, 1:]

                    offset = batch_regress + 15  # 距离左端点-15的偏移量
                    index_float = offset / 0.75  # 浮点数索引
                    index_int = torch.round(index_float).long() #四舍五入
                    index_int = torch.clamp(index_int, 0, 30)
                    batch_conf = torch.gather(batch_conf, dim=2, index=index_int)  # [B, T, K]

                    batch_regress = batch_regress + batch_mean

                    confidence_list.append(batch_conf)
                    predictions_list.append(batch_regress)

            confidence = torch.stack(confidence_list, dim=0)  # [total_len]
            predictions = torch.cat(predictions_list, dim=0)

        if predictions.device.type == 'cuda':
            confidence=confidence.cpu().numpy().reshape(-1)[:]
            predictions = predictions.cpu().numpy().reshape(-1)[:]
            labels =labels.cpu().numpy().reshape(-1)[:]

        '''区间图'''
        plt.figure(figsize=(40, 10))
        plt.plot(labels[:], label='Truth', color='black', linestyle='-', linewidth=2)

        # 计算预测值的上下边界（±2）
        predictions_upper = predictions + 1.5  # 上边界
        predictions_lower = predictions - 1.5  # 下边界

        # 绘制上下边界的填充区域（半透明，边缘加深）
        plt.fill_between(
            np.arange(len(predictions)),  # x轴范围
            predictions_lower,  # 下边界
            predictions_upper,  # 上边界
            color='red',  # 填充区域颜色
            alpha=0.3,  # 填充透明度
            edgecolor='darkred',  # 边缘颜色（加深为暗红色）
            linewidth=2,  # 边缘线宽（适当加粗）
            label='Prediction ±2 Range'
        )

        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.title(use_model_name, size=20)
        plt.legend(fontsize=25)
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=1.2, alpha=0.6)
        plt.tight_layout()
        plt.show()
        '''散点图'''
        # 绘制散点：x为真实标签，y为预测值；用置信度映射颜色（置信度越高，颜色越绿）
        scatter = plt.scatter(labels, predictions, c=confidence, cmap='Greens', alpha=0.8)

        # 添加参考线（y=x，代表预测完全准确的情况）
        min_val = min(np.min(labels), np.min(predictions))
        max_val = max(np.max(labels), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

        # 计算并显示皮尔逊相关系数（PCC）
        pcc = np.corrcoef(labels, predictions)[0, 1]
        plt.text(0.05, 0.9, f'PCC = {pcc:.4f}', transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))

        # 配置图表
        plt.colorbar(scatter, label='Confidence')
        plt.xlabel('Actual Value/deg')
        plt.ylabel('Forecasting Value/deg')
        plt.title('Prediction vs Actual Value')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')  # 将图例放在左下角
        plt.show()

        '''具体值'''
        plt.figure(figsize=(35, 12))
        counter = 0
        plt.plot(labels[:], label='Truth', color='black', linestyle='-', linewidth=3)

        for i in range(0, len(predictions), INPUT_STEPS + OUT_STEPS):
            end = i + OUT_STEPS
            if end > len(predictions):
                end = len(predictions)

            x_vals = np.arange(i, end)
            y_vals = predictions[i:end]

            # 绘制预测曲线
            plt.plot(
                x_vals,
                y_vals,
                linestyle='--',
                color='red',
                linewidth=3,
                marker='o',
                markersize=3,
                markevery=20,
                label='Prediction' if i == 0 else ""
            )

            # # 智能标注 confidence
            # if counter < len(confidence):
            #     last_y = None
            #     for j, (x, y) in enumerate(zip(x_vals, y_vals)):
            #         if counter + j >= len(confidence):
            #             break
            #
            #         conf_val = confidence[counter + j] * 100  # 转换为百分数
            #         # 自动避让
            #         if last_y is not None and abs(y - last_y) < 0.2 * np.std(y_vals):
            #             offset = 0.3 * np.std(y_vals)
            #             va = 'bottom' if (j % 2 == 0) else 'top'
            #             y_text = y + offset if va == 'bottom' else y - offset
            #         else:
            #             y_text = y
            #             va = 'bottom'
            #         last_y = y
            #
            #         plt.text(
            #             x,
            #             y_text,
            #             f"{conf_val:.1f}%",  # 显示一位小数的百分数
            #             ha='center',
            #             va=va,
            #             fontsize=16,
            #             color='blue',
            #             alpha=0.9
            #         )
            #
            # counter += len(x_vals)
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.title(use_model_name, size=20)
        plt.legend(fontsize=25)
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=2, alpha=0.8)
        plt.tight_layout()
        plt.show()


class EarlyStopping:
    def __init__(self, min_delta=0, patience=30):   #delta可容许变化量没用到，patience是连续不间断的超过2次才使counter触发
        self.min_delta = min_delta
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
        if val_loss > (self.min_loss + self.min_delta):
            self.counter += 1

        return self.counter >= self.patience


def compile_and_fit(model, window, max_epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)  # 加入 L2 正则化
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)  #分类任务最大化准确率
    model.compile(
        value_loss=nn.MSELoss(),
        conf_loss=nn.KLDivLoss(reduction='batchmean'),
        align_loss=nn.MSELoss(),
        metric_fn=torchmetrics.Accuracy(task='multiclass',num_classes=100),
        optimizer=optimizer,
        scheduler=scheduler

    )
    early_stopping = EarlyStopping()
    best_val_loss =np.inf
    train_loss_plot = []
    val_loss_plot = []


    for t in range(max_epochs):
        loss, metric = model.train_epoch(window.train)   #调用上面Model
        val_loss, val_metric = model.evaluate(window.val)
        scheduler.step(val_loss)  #根据验证集loss更新学习率调度器
        current_lr = optimizer.param_groups[0]['lr']          # 获取当前学习率


        info = ' - '.join([
            f'[Epoch {t + 1}/{max_epochs}]',
            f'loss: {loss:.4f}',
            f'metric: {metric:.4f}',
            f'val_loss: {val_loss:.4f}',
            f'val_metric: {val_metric:.4f}',
            f'lr: {current_lr:.6f}'
        ])
        print(info)

        train_loss_plot.append(loss)
        val_loss_plot.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights=model.state_dict()
            torch.save(best_weights, r'E:\machine learning\Pt\best.pt')

        if early_stopping(val_loss):
            print('Early stopping')
            plt.figure(figsize=(30,10))
            plt.plot(train_loss_plot,label='Train_loss', color='blue', marker='x')    #训练损失函数图
            plt.plot(val_loss_plot,label='val_loss', color='red', marker='o')
            plt.legend()
            plt.show()
            torch.save(model.state_dict(), r'E:\machine learning\Pt\last.pt')
            break



def test_evaluate(model, window):
    inputs_list =[]
    predictions_list = []
    with torch.no_grad():
        for inputs, labels in window.test:  # 遍历 DataLoader 中的每个批次,_为标签数据不需要


            batch = inputs.to(device)  # 将输入数据移动到设备上
            # 修改输入数据格式
            batch_mean = batch.mean(dim=1, keepdim=True)
            batch = batch - batch_mean  # 中心化
            batch_pred = model(batch, return_softmax=True)

            batch_pred = batch_pred[:,:,0].unsqueeze(-1)

            batch_pred = batch_pred.unsqueeze(-1) + batch_mean
            predictions_list.append(batch_pred.view(-1).cpu())


            labels=labels[:,:,0].unsqueeze(-1)
            inputs_list.append(labels.view(-1))

    truth = torch.cat(inputs_list, dim=0).numpy()
    predict = torch.cat(predictions_list, dim=0).numpy()


    # 评价指标
    mse = np.mean((predict - truth) ** 2)
    print('mse', mse)
    rmse = np.sqrt(np.mean((np.array(truth) - np.array(predict)) ** 2))
    print('rmse', rmse)



#窗口
INPUT_STEPS=10
OUT_STEPS = 10   #影响训练和预测的步数，因为在模型中输出步数就固定了
# num_features=df.shape[1]
multi_window = WindowGenerator(
    input_width=INPUT_STEPS,          #input只与训练有关
    label_width=OUT_STEPS,   #OUT_STEPS应该与label_width相同，应为训练过程预测到的OUT_STEPS是与label_width比较
    shift=OUT_STEPS                  #shift只与训练有关
)

from Models import CNN_LSTM_Classify,TCN_LSTM_Classify,LSTM_Classify

#训练
device = torch.device("cuda")
use_model =LSTM_Classify.LSTM_Detect(pred_len=OUT_STEPS,num_classes=40)
compile_and_fit(use_model.to(device), multi_window)



#预测
state_dict=torch.load(r'E:\machine learning\Pt\best.pt')
use_model.load_state_dict(state_dict)
use_model_name=type(use_model).__name__
print('预测模型：',use_model_name,
      '输入 输出长度：',INPUT_STEPS,OUT_STEPS,
      )


use_model.eval()
multi_window.plot(use_model)
test_evaluate(use_model.to(device),multi_window)





