'''真实数据,采样频率0.05s/20Hz，周期2s'''
'''只用来测试训练好的权重在其他数据集上的画图效果'''
'''用的分类方法'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv(r"E:\shiyan\ShipMotion_Classify\8-Still-YNY-20221212-1045-one-hot-30bins.csv")
plot_cols = df.filter(regex='^phi_').columns.tolist()
if 'phi' not in plot_cols:
    plot_cols.insert(0, 'phi')  # 将'phi'添加到列名列表的开头
df = df[plot_cols]
#采样频率降为0.1s
df=df[::2]

train_df =df
val_df = df
test_df = df



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
        inputs=inputs[:,:,:]    #取高斯编码后的所有列


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
                    batch_pred = model(batch,return_softmax=True)

                    conf = torch.max(batch_pred, dim=2).values  # shape [1, seq_len] 或 [seq_len]
                    print(conf)
                    conf_mean = conf.mean()  # 标量 tensor
                    confidence_list.append(conf_mean)  # 可以 append 标量 tensor

                    max_indices = torch.argmax(batch_pred, dim=2)
                    batch_pred = -15+max_indices          #最大概率对应的索引位置

                    predictions_list.append(batch_pred)

            confidence = torch.stack(confidence_list, dim=0)  # [total_len]
            predictions = torch.cat(predictions_list, dim=0)

        if predictions.device.type == 'cuda':
            confidence=confidence.cpu().numpy().reshape(-1)
            predictions = predictions.cpu().numpy().reshape(-1)[:2000]
            labels =labels.cpu().numpy().reshape(-1)[:2000]

        plt.figure(figsize=(40, 10))

        counter = 0
        plt.plot(labels, label='真实值', color='black', linestyle='-', linewidth=2)
        for i in range(0, len(predictions), INPUT_STEPS + OUT_STEPS):
            end = i + OUT_STEPS
            if end > len(predictions):
                end = len(predictions)  # 防止越界

            x_vals = np.arange(i, end)
            y_vals = predictions[i:end]

            # 画预测曲线
            plt.plot(
                x_vals,
                y_vals,
                linestyle='--',
                color='red',
                linewidth=3,
                marker='o',
                markersize=3,
                markevery=20,
                label='预测值' if i == 0 else ""
            )

            # 在每段起始点标注 confidence
            # if counter < len(confidence):  # 防止越界
            #     plt.text(
            #         x_vals[0],  # 起点横坐标
            #         y_vals[0],  # 起点纵坐标
            #         f"{confidence[counter]:.2f}",  # 显示保留两位小数
            #         ha='center', va='bottom',
            #         fontsize=20, color='blue'
            #     )
            # counter += 1

        plt.title(use_model_name)
        plt.legend()
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




def test_evaluate(model, window):
    inputs_list =[]
    predictions_list = []
    with torch.no_grad():
        for inputs, labels in window.test:  # 遍历 DataLoader 中的每个批次,_为标签数据不需要


            inputs = inputs.to(device)  # 将输入数据移动到设备上
            batch_predictions = model(inputs,return_softmax=True)  # 对当前批次进行预测
            max_indices = torch.argmax(batch_predictions, dim=2)
            batch_pred = -15 + max_indices
            predictions_list.append(batch_pred.view(-1).cpu())  # 将预测结果添加到列表中

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
num_features=df.shape[1]
multi_window = WindowGenerator(
    input_width=INPUT_STEPS,          #input只与训练有关
    label_width=OUT_STEPS,   #OUT_STEPS应该与label_width相同，应为训练过程预测到的OUT_STEPS是与label_width比较
    shift=OUT_STEPS                  #shift只与训练有关
)

from Models import CNN_LSTM_Classify,TCN_LSTM_Classify

device = torch.device("cuda")
use_model =TCN_LSTM_Classify.TCN_LSTM_Classify(pred_len=OUT_STEPS,num_classes=30)



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






