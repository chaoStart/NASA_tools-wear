from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  ## 用来正常显示负号
import pandas as pd
# 测试一下
for i in range(1, 6):
    if i == 3:
        # continue  # 当 i 等于 3 时，跳过本次循环的剩余代码
        break  # 当 i 等于 3 时，结束所有循环
    print(f"Current value of i: {i}")

#
mat_path = "./NASA_mill/mill.mat"
raw_data = loadmat(mat_path)  # type: dict
# 查看文件中的变量
print(raw_data.keys())
print('******************')
mill_data = raw_data['mill']
print('读取指定mill变量的类型:',type(mill_data))
print('读取指定mill变量的形状:',mill_data.shape)
# 去掉多余的一个维度，使得（1,167）-->(167,)
mill_1=mill_data[0]
print('读取指定mill变量第一个数组的类型:',type(mill_1),mill_1.shape)
print('------------------------------------')
# 读取（167，）中的第94个数据;可以看见第94组的采集样本数据有异常，因此后续剔除了该组数据
print('读取指定mill_1,(167,)变量第94个数组的形状:\n',mill_1[94])
# 读取（167，）中的第3个数据
print('读取指定mill_1,(167,)变量第3个数组的形状:\n',mill_1[2])
# 读取（167，）中的第4个数据
print('读取指定mill_1,(167,)变量第4个数组的形状:\n',mill_1[3])
print('读取指定mill变量第4个数组VB的形状:',mill_1[3][2].shape)
print('读取指定mill_1的形状:',mill_1[3][2])
# mill_1[3][2]是一个2维度的多余维度数据，mill_1[3][2][0][0]可以直接读取数值
print('读取（167，）中第4组数据的VB磨损值:',mill_1[3][2][0][0])
print('------------------------------------')
VB=[]
for i in  range(167):
    VB.append(mill_data[0][i][2][0][0])
print("打印磨损值VB的类型:",type(VB))
VB=np.array(VB)
print("转化为Numpy数组后的类型",type(VB))
print("数组的长度",len(VB))
print("打印磨损值VB的具体数值:\n",VB)
# 画出曲线图
X = np.arange(1, 168)  # 横坐标 X 从1到167
Y = VB  # 替换为你的数据数组，磨损数据有167个数据
plt.plot(X, Y)
# 添加标题和坐标轴标签
plt.title('曲线图')
plt.xlabel('X轴')
plt.ylabel('Y轴')
# 显示图形
plt.show()
print('------------------------------------')
print('读取指定mill_1变量第1个数组里面的AC电流变量:',type(mill_1[0][7]),mill_1[0][7].shape)
print('读取指定mill_1变量第1个数组里面的AE声音变量:',type(mill_1[0][12]),mill_1[0][12].shape)