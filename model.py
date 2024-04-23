from keras.models import Model,Input
from keras.layers.convolutional import Conv1D,MaxPooling1D
# from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization,GlobalMaxPooling1D
from keras.layers.core import Dense,Dropout,Flatten,Activation
from keras.layers.merge import concatenate,add
from keras.optimizers import *
from keras.optimizers import adam_v2
from keras.layers.advanced_activations import LeakyReLU

def repeated_block(x,filters,kernel_size=3,pooling_size=3,dropout=0.5,
                   is_first_layer_of_block=False,activation=LeakyReLU()):
    """
    residual block using pre activation
    :param x:
    :param filters:
    :param kernel_size:
    :param pooling_size:
    :param dropout:
    :param is_first_layer_of_block:
    :return:
    """
    k1,k2 = filters
    # program control it
    conv1 = Conv1D(k1,kernel_size,strides=1,padding='same')(x)
    out = BatchNormalization()(conv1)
    out = Activation(activation)(out)
    out = Dropout(dropout)(out)

    conv2 = Conv1D(k2,kernel_size,strides=1,padding='same')(out)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = Dropout(dropout)(conv2)
    print('conv2的',conv2.shape)
    # if is_first_layer_of_block:
    #     # add conv here
    #     pooling = Conv1D(k2,kernel_size,strides=1,padding='same')(x)
    # else:
    #     pooling = MaxPooling1D(pooling_size, strides=2, padding='same')(x)
    #     pass
    pooling = MaxPooling1D(pooling_size, strides=1, padding='same')(x)
    residual = Conv1D(k2, 1, strides=1, padding='same')(pooling)  # Match shapes for residual connection
    print('residual的',residual.shape)
    output = add([conv2,residual])
    return output
def build_residual_model(signal_timestep,
                         signal_dimension,
                         catalog,
                         number_conf,
                         output_dim,
                         block_number=15,
                         dropout=0.2,
                         activation='relu'):
    '''
    build residual neural network for NASA data
    :param signal_timestep:
    :param signal_dimension:
    :param catalog:
    :param number_conf:
    :param activation:
    :return:
    '''
    # (166,9000,6)    (166,2)   (166,3)
    signal_input = Input(shape=(signal_timestep,signal_dimension),name='signal_input')
    catalog_input = Input(shape=(catalog,),name='material_one_hot_input')
    number_input = Input(shape=(number_conf,),name='conf_number')
    print('signal_input的形状:',signal_input.shape)
    block_part_num = 5
    base_filter = 8
    output = signal_input
    total_times = block_number // block_part_num   ## 15//5=3
    for cur_layer_num in range(block_number):
        is_first_layer = False
        # if cur_layer_num % block_part_num == 0:
        #     is_first_layer = True
        # determine kernel size
        filter_times = total_times - cur_layer_num // block_part_num #3-x//5=y
        # filter = (base_filter*(2**(filter_times)),base_filter*(2**(filter_times)))#[8*(2**y),8*(2**y)]=(64,64)
        filter = (64,128)#(16,16)
        print('打印filter,block_number=15,cur_layer_num=(0~14)')
        print(filter,block_number,cur_layer_num)
        # (166-->9000,6)  #[8*(2*y),8*(2*y)]  #dropout=0.1 #is_first_layer=False #activation='relu'
        output = repeated_block(output, filter, dropout=dropout,is_first_layer_of_block=is_first_layer,activation=activation)

    print('out的形状:',output.shape)
    output = BatchNormalization()(output)
    print('BatchNormalization之后out的形状:',output.shape)
    output = Flatten()(output)
    # 全连接层
    output = Dense(64, activation=activation)(output)
    output = Dropout(dropout)(output)
    output = Activation(activation)(output)
    # 输出层
    output = Dense(output_dim,activation='linear')(output)
    # model = Model(inputs=[signal_input,catalog_input,number_input],outputs=output)
    model = Model(inputs=signal_input,outputs=output)
    optimizer = adam_v2.Adam(learning_rate=1e-2)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
    return model