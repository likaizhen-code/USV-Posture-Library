'''用来生成分类数据集'''
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = '14-Still-20221212-1603'
df = pd.read_csv(rf'E:\shiyan\ShipMotion_Real\{dataset}.csv')

#%% 打标签
def map_value_to_label(value, min_val=-15, max_val=15, num_bins=30):
    """将数值映射到0-99的标签"""
    if value >= max_val:
        return num_bins - 1  # 最大值15单独作为标签99
    elif value < min_val:
        return 0  # 小于-15的值映射到标签0（可根据需要调整）

    # 计算区间宽度
    bin_width = (max_val - min_val) / num_bins
    # 计算标签索引
    label = int((value - min_val) / bin_width)
    return label

# 对目标列应用映射函数，创建新的标签列
df['label'] = df['phi'].apply(map_value_to_label)

# 保存
df.to_csv(fr'E:\shiyan\ShipMotion_Classify\{dataset}.csv', index=True)

#%% 高斯平滑处理标签
def gaussian_smooth_label(label, num_bins=100, sigma=1.0):
    """生成高斯平滑后的标签分布（概率分布），中心在 label（可以是小数）"""
    labels = np.arange(num_bins)
    distances = labels - label
    smoothed_probs = np.exp(-0.5 * (distances / sigma) ** 2)
    smoothed_probs /= np.sum(smoothed_probs)
    return smoothed_probs

# -------- 应用高斯平滑 --------
def apply_smoothing(df, label_col='label', num_bins=100, sigma=1.0):
    smoothed = np.vstack([
        gaussian_smooth_label(label, num_bins=num_bins, sigma=sigma)
        for label in df[label_col]
    ])

    smoothed_df = pd.DataFrame(
        smoothed,
        columns=[f'phi_smooth_{i}' for i in range(num_bins)]
    )

    return pd.concat([df.reset_index(drop=True), smoothed_df], axis=1)

# df_smooth = apply_smoothing(df, label_col='label', num_bins=100, sigma=0.3)
# df_smooth.to_csv(f'E:/shiyan/ShipMotion_Classify/{dataset}.csv', index=False)


#%% 独热编码
def one_hot_encoding(label, num_bins=100):
    """生成独热编码：指定位置为1，其余为0"""
    one_hot = np.zeros(num_bins)
    # 确保标签在有效范围内
    label = np.clip(label, 0, num_bins - 1)
    one_hot[int(label)] = 1
    return one_hot
# -------- 应用独热编码 --------
def apply_one_hot(df, label_col='label', num_bins=100):
    one_hot_labels = np.vstack([
        one_hot_encoding(label, num_bins=num_bins)
        for label in df[label_col]
    ])

    one_hot_df = pd.DataFrame(
        one_hot_labels,
        columns=[f'phi_onehot_{i}' for i in range(num_bins)]
    )

    return pd.concat([df.reset_index(drop=True), one_hot_df], axis=1)

# 应用独热编码（替换高斯平滑步骤）
df_onehot = apply_one_hot(df, label_col='label', num_bins=30)
df_onehot.to_csv(f'E:/shiyan/ShipMotion_Classify/{dataset}-one-hot-30bins.csv', index=False)












