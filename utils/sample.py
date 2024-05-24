import pandas as pd


input = 'dataset/Hainan_A5_2cm_train.csv'
output = 'dataset/A5N15.csv'
num = 69

# 读取数据
data = pd.read_csv(input, header=None)
# 随机采样得到 num 个样本
data = data.sample(n=num)
# 写入数据
data.to_csv(output, index=False, header=False)








