from numpy.core.multiarray import ndarray
from model import build_residual_model
from data import DataSet
from model1 import build_multi_input_main_residual_network
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

# PREDICT = False
PREDICT = True
LOG_DIR = './regression_log/'
DROPOUT = 0.3
###下面进入模型进行预测#####
for depth in [20,15,20,10]:
    print('depth的数量：',depth)
    train_name = 'regression_depth_%s'%(depth)
    model_name = '%s.kerasmodel' % (train_name)
    weight_name = '%s.keras_weight'%(train_name)
    if not PREDICT:
        #(166,9000,6)    (166,2)   (166,3)
        tb_cb = TensorBoard(log_dir=LOG_DIR+train_name)
        model = build_multi_input_main_residual_network(signal_input.shape[1],
                                                        signal_input.shape[2],
                                                        catalog_input.shape[1],
                                                        number_input.shape[1],
                                                        1,block_number=depth,dropout=DROPOUT)
        print('y的形状:',y.shape)
        print('model has been established')
        model.fit([signal_input,catalog_input,number_input],y,callbacks=[tb_cb],batch_size=16,shuffle=False,epochs=200,validation_split=0.2)
        model.summary()
        # model.save(model_name)
        model.save("20_informationmodel.h5")
        print('模型保存成功ending~~')
        # 选择一个深度Depth=? 进行训练，结束后不再进行训练
        break
    else:
        from keras.models import load_model
        # model = load_model(model_name)
        model = load_model("20_informationmodel.h5")
        y_pred = model.predict([signal_input,catalog_input,number_input])
        if model_name=='regression_depth_20.kerasmodel':
            np.save('regression_depth_20', y_pred)
        print('y_pred的数据是:\n', y_pred.shape)
        y_pred = np.squeeze(y_pred)
        print('去掉1个维度后y_pred的形状', y_pred.shape)
        print('真实y的形状', y.shape)
        # X = np.arange(1, 167)  # 横坐标 X 从1到165
        plt.plot(y, label="y", color='pink')
        plt.title('刀具真实值')
        # plt.show()
        plt.plot(y_pred, label="pred", color='blue')
        plt.title('刀具磨损预测结果')
        plt.legend(loc='upper left')
        plt.show()
        # 选择第一个Depth=20时，绘制出来的图像，结束后不再绘图
        break