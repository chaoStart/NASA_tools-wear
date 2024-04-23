from numpy.core.multiarray import ndarray
from model import build_residual_model
from data import DataSet
import time
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  ## 用来正常显示负号

data = DataSet()
# original data may have data missing, using random forest to predict that
y = np.array(data.rf_vb_value)
print('y的形状:',y.shape)

# remove dirty data
y = np.delete(y,17)
print('处理之后的y的形状',y.shape)
# input('Press Enter To Continue')
for index,value in enumerate(y):
    print('&',index,value)
# X = np.arange(1, 167)  # 横坐标 X 从1到167
plt.plot(y,label="y value",color='black',linestyle='--')
plt.title('刀具磨损实际结果')
plt.legend(loc='upper left')
plt.show()
signal_input,catalog_input,number_input = data.signal_value,data.material_type,data.number_value # type:#(ndarray, ndarray)
signal_input,catalog_input,number_input = np.delete(signal_input,17,axis=0),np.delete(catalog_input,17,axis=0),\
                                          np.delete(number_input,17,axis=0)
#   (166,9000,6)  (166,2)   (166,3)
print(signal_input.shape,catalog_input.shape,number_input.shape)



# 下面的代码是由于因为batch_size=32时，导致内存溢出OOM发生，因此使用生成器逐次加载数据
X_all,y_all =signal_input,y
print('x_all和y_all的数据形状大小:',type(X_all),X_all.shape,type(y_all),y_all.shape)
# Split the data into training and validation sets
X_train_signal, X_val_signal, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, shuffle=False  # Adjust test_size and shuffle as needed
)
print('x_train_signal和y_train的数据形状大小:',X_train_signal.shape,y_train.shape)
window_size = 1
train_generator = TimeseriesGenerator(
    data=X_train_signal,
    targets=y_train,
    length=window_size,
    sampling_rate=1,
    batch_size=32
)
# window_size1 = 32
val_generator = TimeseriesGenerator(
    data=X_val_signal,
    targets=y_val,
    length=window_size,
    sampling_rate=1,
    batch_size=32
)

# PREDICT = False
PREDICT = True
LOG_DIR = './regression_log/'
DROPOUT = 0.3
###下面进入模型进行预测#####
for depth in [15,20,10]:
    print('depth的数量：',depth)
    train_name = 'regression_depth_%s_%s_relu'%(depth,DROPOUT)
    model_name = '%s.kerasmodel' % (train_name)
    weight_name = '%s.keras_weight'%(train_name)
    if not PREDICT:
        #(166,9000,6)    (166,2)   (166,3)
        tb_cb = TensorBoard(log_dir=LOG_DIR+train_name)
        model = build_residual_model(signal_input.shape[1],
                                     signal_input.shape[2],
                                     catalog_input.shape[1],
                                     number_input.shape[1],
                                     1,
                                     block_number=depth,dropout=DROPOUT,activation='relu')

        print('model has been established')
        # model.fit([signal_input,catalog_input,number_input],y,callbacks=[tb_cb],batch_size=1,shuffle=False,epochs=60,validation_split=0.2)
        batch_size = 16
        generator = TimeseriesGenerator(X_all, y_all, length=window_size, sampling_rate=1, batch_size=batch_size)
        # 划分训练集和验证集的生成器
        train_size = int(len(generator) * 0.8)
        train_generator = TimeseriesGenerator(X_all[:train_size], y_all[:train_size], length=window_size,
                                              sampling_rate=1, batch_size=batch_size)
        val_generator = TimeseriesGenerator(X_all[train_size:], y_all[train_size:], length=window_size,
                                            sampling_rate=1, batch_size=batch_size)
        # 将验证集转换为 NumPy arrays
        X_val, y_val = [], []
        for i in range(len(val_generator)):
            x_batch, y_batch = val_generator[i]
            X_val.append(x_batch)
            y_val.append(y_batch)
        X_val = np.concatenate(X_val)
        # 使用 np.squeeze 去掉大小为1的维度
        X_val = np.squeeze(X_val)
        y_val = np.concatenate(y_val)
        print('X_val和y_val的数据集:',X_val.shape,y_val.shape)
        model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=60,
                            validation_data=val_generator, validation_steps=len(val_generator))
        # model.fit_generator(
        #     train_generator,
        #     steps_per_epoch=len(train_generator),
        #     epochs=60,
        #     validation_data=val_generator,
        #     validation_steps=len(val_generator)
        # )
        model.summary()
        model.save(model_name)
        # model.save("Kerasmodel.h5")
        print('模型保存成功ending~~')
        break

    else:
        from keras.models import load_model
        # model = load_model(model_name)
        model = load_model("Kerasmodel.h5")
        y_pred = model.predict([signal_input,catalog_input,number_input])
        print('y_pred的数据是:\n',y_pred)
        y_pred=np.squeeze(y_pred)
        print('预测后y_pred的形状', y_pred.shape)
        print('处理之后真实y的形状', y.shape)
        # X = np.arange(1, 167)  # 横坐标 X 从1到165
        plt.plot(y,label="y",color='pink')
        plt.title('刀具真实值')
        plt.show()
        plt.plot(y_pred,label="pred",color='blue')
        plt.title('刀具磨损预测结果')
        plt.legend(loc='upper left')
        plt.show()
        break