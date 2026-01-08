'''真实数据,采样频率0.05s/20Hz，周期2s'''
'''回归+分类+loss可视化+自适应！！！'''
'''集成好的自适应模块'''
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

    def plot(self, model=None):
        # 计算一个完整的、不重叠的序列长度 (输入20步 + 输出10步)
        sequence_len = INPUT_STEPS + OUT_STEPS  # 30

        # 1. 保证 test_df 行数是 sequence_len (30) 的倍数，多余部分截掉
        total_len = len(self.test_df)
        new_len = total_len - (total_len % sequence_len)
        self.test_df = self.test_df.iloc[:new_len]

        N, F = self.test_df.shape  # N: 截断后的总行数, F: 特征数
        B = N // sequence_len  # B: 不重叠的批次数/序列数

        # --- 数据准备 ---
        # Reshape: (B, sequence_len, F)
        data_reshaped = self.test_df.values.reshape(B, sequence_len, F)

        # Inputs: (B, INPUT_STEPS, F)
        inputs_np = data_reshaped[:, :INPUT_STEPS, :]

        # Labels (True Target): (B, OUT_STEPS, F)
        labels_np_target = data_reshaped[:, INPUT_STEPS:, :]

        # 完整标签 (Full Label): N 行 F 列，用于绘制背景
        # 我们只绘制第一个特征 (假设 F=1 或只关注第一个特征)
        labels = self.test_df.values[:, 0]

        # 转换为 PyTorch Tensor 并移至设备
        inputs = torch.tensor(inputs_np, dtype=torch.float).to(device)

        # --- 模型推理 (Inference) ---
        predictions = None
        if model is not None:
            model.eval()
            model.to(device)
            predictions_list = []
            confidence_list=[]

            with torch.no_grad():
                for i in range(B):  # 遍历所有批次/序列
                    batch = inputs[i].unsqueeze(0)

                    batch_mean =batch.mean(dim=1, keepdim=True)
                    batch_pred = model(batch,return_softmax=True)   #一定要加不然会取负数

                    batch_regress = batch_pred[:, :, 0].unsqueeze(-1)
                    batch_conf = batch_pred[:, :, 1:]

                    offset = batch_regress -batch_mean+ 15  # 距离左端点-15的偏移量，一定注意：概率值取去均值后的！！！！！！！
                    index_float = offset / 0.75  # 浮点数索引
                    index_int = torch.round(index_float).long()  # 四舍五入
                    index_int = torch.clamp(index_int, 0, 30)
                    batch_conf = torch.gather(batch_conf, dim=2, index=index_int)  # [B, T, K]

                    confidence_list.append(batch_conf)
                    predictions_list.append(batch_regress)

            # predictions shape: (B, OUT_STEPS, F)
            confidence = torch.stack(confidence_list, dim=0)  # [total_len]
            predictions = torch.cat(predictions_list, dim=0)

            if predictions.device.type == 'cuda':
                confidence = confidence.cpu().numpy().reshape(-1)[:]
                predictions = predictions.cpu().numpy()

        if predictions is not None:

            '''散点图'''
            plt.figure(figsize=(8, 6))
            labels_true = labels_np_target[:, :, 0].reshape(-1)  # (B*OUT_STEPS,)
            predictions_flat = predictions[:, :, 0].reshape(-1)
            confidence_flat = confidence.reshape(-1)
            # 绘制散点：x为真实标签，y为预测值；用置信度映射颜色（置信度越高，颜色越绿）
            scatter = plt.scatter(labels_true, predictions_flat, c=confidence_flat, cmap='Greens', alpha=0.8)

            # 添加参考线（y=x，代表预测完全准确的情况）
            min_val = min(np.min(labels_true), np.min(predictions_flat))
            max_val = max(np.max(labels_true), np.max(predictions_flat))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

            # 计算并显示皮尔逊相关系数（PCC）
            pcc = np.corrcoef(labels_true, predictions_flat)[0, 1]
            plt.text(0.05, 0.9, f'PCC = {pcc:.4f}', transform=plt.gca().transAxes, fontsize=16,
                     bbox=dict(facecolor='white', alpha=0.8))
            # 配置图表
            cbar = plt.colorbar(scatter, label='Confidence')  # 先创建 colorbar
            cbar.ax.tick_params(labelsize=18)  # 再放大刻度数字
            cbar.set_label('Confidence', fontsize=18,fontweight='bold')  # 放大标签

            plt.xlabel('Actual Value/deg', fontsize=18, fontweight='bold')
            plt.ylabel('Forecasting Value/deg', fontsize=18, fontweight='bold')

            plt.tick_params(axis='both', labelsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower right', fontsize=16)  # 将图例放在左下角
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)  # 坐标轴脊线加粗

            plt.tight_layout()
            plt.show()

            '''回归值'''
            plt.figure(figsize=(35, 12))
            # 1. 绘制完整的真实值 (Label)
            # x_index 范围从 0 到 N-1
            x_full_index = range(N)
            plt.plot(x_full_index, labels, label='Truth', color='black', linestyle='-', linewidth=3)

            # 2. 绘制预测值 (Prediction)
            # 预测结果 predictions_np shape: (B, OUT_STEPS, F)

            for i in range(B):
                # 计算当前批次预测值在完整时间线上的起始 x 坐标
                # 起始位置 = i * (INPUT_STEPS + OUT_STEPS) + INPUT_STEPS
                # 预测值总是从输入序列 (20步) 之后开始
                start_x = i * sequence_len + INPUT_STEPS

                # 预测值对应的 x 坐标范围
                x_pred = range(start_x, start_x + OUT_STEPS)

                # 当前批次的预测值 (只取第一个特征)
                y_pred = predictions[i, :, 0]

                # 绘制预测段
                if i == 0:
                    # 第一次绘制时添加标签
                    plt.plot(x_pred, y_pred,
                             linestyle='--',
                             color='red',
                             linewidth=3,
                             marker='o',
                             markersize=3,
                             markevery=20,
                             label='Prediction' if i == 0 else "")
                else:
                    plt.plot(x_pred, y_pred,
                             linestyle='--',
                             color='red',
                             linewidth=3,
                             marker='o',
                             markersize=3,
                             markevery=20,
                             label='Prediction' if i == 0 else "")

                # 可选：添加垂直线和高亮区域来区分序列
                plt.axvline(x=start_x, color='gray', linestyle=':', alpha=0.3)

            plt.title(f'不重叠序列预测结果对比 (输入{INPUT_STEPS}步, 预测{OUT_STEPS}步)')
            plt.xlabel('时间步索引')
            plt.ylabel('值')
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
        metric_fn=torchmetrics.Accuracy(task='multiclass',num_classes=40),
        optimizer=optimizer,
        scheduler=scheduler

    )
    early_stopping = EarlyStopping()
    best_val_loss =np.inf
    train_loss_plot = []
    train_value_loss_plot = []
    train_conf_loss_plot = []
    train_align_loss_plot = []
    val_loss_plot = []


    for t in range(max_epochs):
        loss,v_loss,c_loss,a_loss, metric = model.train_epoch(window.train)   #调用上面Model
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
        train_value_loss_plot.append(float(v_loss))
        train_conf_loss_plot.append(float(c_loss))
        train_align_loss_plot.append(float(a_loss))
        val_loss_plot.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights=model.state_dict()
            torch.save(best_weights, r'E:\machine learning\Pt\best.pt')

        if early_stopping(val_loss):
            print('Early stopping')
            plt.figure(figsize=(30,10))
            plt.plot(train_loss_plot,label='Train_loss', color='blue', marker='x')    #训练损失函数图
            plt.plot(train_value_loss_plot, label='value_loss')
            plt.plot(train_conf_loss_plot, label='conf_loss')
            plt.plot(train_align_loss_plot, label='align_loss')
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

            batch_pred = model(batch, return_softmax=True)
            batch_pred = batch_pred[:,:,0].unsqueeze(-1)

            batch_pred = batch_pred.unsqueeze(-1)
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
    mae = np.mean(np.abs(predict - truth))
    print('mae', mae)



#窗口
INPUT_STEPS=20
OUT_STEPS = 20   #影响训练和预测的步数，因为在模型中输出步数就固定了
num_features=df.shape[1]
multi_window = WindowGenerator(
    input_width=INPUT_STEPS,          #input只与训练有关
    label_width=OUT_STEPS,   #OUT_STEPS应该与label_width相同，应为训练过程预测到的OUT_STEPS是与label_width比较
    shift=OUT_STEPS                  #shift只与训练有关
)

from Models import LSTM_Classify,CNN_Classify,BiLSTM_Classify


#训练
device = torch.device("cuda")
use_model =BiLSTM_Classify.BiLSTM_Classify(input_len=INPUT_STEPS,pred_len=OUT_STEPS,num_classes=40)
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





