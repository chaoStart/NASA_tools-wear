from keras.layers import Dense,Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.models import Input,Sequential,Model
# from keras.optimizers import Adam,Nadam
from keras.optimizers import adam_v2
# def build_simple_rnn_model(timestep,input_dim,output_dim,dropout=0.4,lr=0.001):
def build_simple_rnn_model(timestep,input_dim,output_dim,dropout=0.4,learning_rate=0.001):
    input = Input((timestep,input_dim))
    # LSTM, Single
    output = LSTM(50,return_sequences=False)(input)
    # for _ in range(1):
    #     output = LSTM(32,return_sequences=True)(output)
    # output = LSTM(50,return_sequences=False)(output)
    output = Dropout(dropout)(output)
    output = Dense(output_dim)(output)
    model =  Model(inputs=input,outputs=output)
    # optimizer = Adam(lr=lr)
    optimizer = adam_v2.Adam(learning_rate=learning_rate)
    model.compile(loss='mae',optimizer=optimizer,metrics=['mse'])
    return model
from keras.layers import merge
from keras.layers.merge import add,concatenate
from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Conv1D,MaxPooling1D
from keras.layers.core import Dense,Activation,Flatten,Dropout,Masking
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model
from keras.layers import Input,TimeDistributed
from keras.layers.recurrent import LSTM

def first_block(tensor_input,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters
    out = Conv1D(k1,1,padding='same')(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)
    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(tensor_input)
    out = add([out,pooling])
    return out
def repeated_block(x,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters
    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)
    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(x)
    out = add([out, pooling])
    return out
def build_multi_input_main_residual_network(
                                signal_timestep,
                                signal_dimension,
                                catalog,
                                number_conf,
                                output_dim,
                                block_number,
                                loop_depth=15,
                                dropout=0.5):

    signal_input = Input(shape=(signal_timestep, signal_dimension), name='signal_input')
    catalog_input = Input(shape=(catalog,), name='material_one_hot_input')
    number_input = Input(shape=(number_conf,), name='conf_number')
    print('signal_timestep,signal_dimension参数值:',signal_timestep,signal_dimension)
    print('catalog参数值:',catalog)
    print('number_conf参数值:',number_conf)
    print('dropout参数值:',dropout)
    print('output_dim参数值:',output_dim)
    print('output的输出值:',signal_input.shape)
    out = Conv1D(128,5)(signal_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    print('out的形状',type(out),out.shape)
    # kernel_size表示卷积核的  大小 为3
    # filters 滤波器(卷积核)的  数量 分别是64、128，赋值给K1，K2
    out = first_block(out,(64,128),dropout=dropout)
    for _ in range(loop_depth):
        out = repeated_block(out,(64,128),dropout=dropout)
    # add flatten
    out = Flatten()(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)
    model = Model(inputs=[signal_input,catalog_input,number_input],outputs=[out])
    optimizer = adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mse','mae'])
    return model