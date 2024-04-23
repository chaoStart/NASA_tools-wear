from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from data import DataSet
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  ## 用来正常显示负号
def fit_value_by_random_forest():
    catalog = ['time', 'DOC', 'feed', 'material']
    data = DataSet()
    # vb_val = data.vb_value
    # vb_pd_val = pd.DataFrame(vb_val)
    df = data.export_as_pd  #将 data 实例的 export_as_pd 方法赋值给名为 df 的变量
    df.dropna()#dropna() 函数通常是用来删除数据框（DataFrame）中包含缺失值（NaN或None）的行或列。
    print("df的具体数据是什么:",type(df))#df是磨损值VB和剩余4个切削工况信息
    print("df的具体数据是什么:\n",df)
    print("********************")
    null_index_list = []
    for index, list in enumerate(df['VB']):
        if math.isnan(list):
            null_index_list.append(index)
        else:
            print(list,type(list))
            pass
    print(null_index_list)
    df_drop = df.drop(null_index_list)
    print("df_drop函数去除掉Nan类型后的数组:\n",df_drop)

    regression = RandomForestRegressor(n_jobs=4, n_estimators=100, oob_score=True)
    regression.fit(df_drop[catalog], df_drop['VB'])
    preds = regression.predict(df[catalog])
    print('预测的补充数据的',preds.shape)
    plt.plot(preds, label='prediction',color='pink')
    plt.plot(df['VB'], label='real',color='blue')
    plt.title('label=原始缺失刀具磨损数据')
    plt.legend(loc='upper left')
    # plt.show()
    print('*********************')
    # fit data
    for index,value in enumerate(df['VB']):
        if math.isnan(value):
            df['VB'][index] = round(preds[index],2)
            null_index_list.append(index)
    print("随机森林函数完善后的数据:\n", df['VB'])
    # 示例数据
    # X = np.arange(1, 167)  # 横坐标 X 从1到166
    # plt.plot(X,df['VB'],label='tool wear',color='red')
    plt.plot(df['VB'],label='tool wear',color='black', linestyle='--')
    plt.title('label=修复缺失后的刀具磨损数据')
    plt.legend(loc='upper left')
    plt.show()
    return df['VB']

if __name__ == '__main__':
    fit_value_by_random_forest()