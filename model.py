# coding: utf-8
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt

from keras.layers import Activation, Dropout, AveragePooling2D, AtrousConvolution2D, ZeroPadding2D, Lambda, multiply
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, Input, UpSampling2D, Conv2DTranspose, Add
from keras.layers import Reshape, Dense
from keras import backend as K
from util import *


def addLayer(previousLayer, nOutChannels):
    # 对上一层 previousLayer 进行批标准化（Batch Normalization），其中 axis=-1 表示在最后一个轴上进行标准化。这有助于加速训练和提高模型的鲁棒性。
    bn = BatchNormalization(axis=-1)(previousLayer)
    # 通过 ReLU（Rectified Linear Unit）激活函数激活批标准化的输出。ReLU 是一种常用的非线性激活函数，它在正数区域返回输入值，而在负数区域返回零。
    relu = Activation('relu')(bn)
    # 应用一个 1x1 的卷积层，用于调整通道数为 nOutChannels。这种 1x1 的卷积通常用于维度变换和特征整合。
    relu = Conv2D(nOutChannels, (1, 1), padding="same")(relu)
    # 再次对卷积层的输出进行批标准化。
    bn_1 = BatchNormalization(axis=-1)(relu)
    # 再次通过 ReLU 激活函数激活批标准化的输出。
    relu_1 = Activation('relu')(bn_1)
    # ：应用一个3x3的卷积层，border_mode='same' 表示使用“same”填充，以保持输入输出的空间尺寸相同。
    conv = Conv2D(nOutChannels, 3, 3, border_mode='same')(relu_1)
    # 通过 Add 层将卷积层的输出与输入层 previousLayer 相加，实现残差连接。这有助于减轻梯度消失问题，促使模型更容易学到恒等映射。
    return Add()([conv, previousLayer])


def addTransition(previousLayer, nOutChannels, dropRate, blockNum):
    bn = BatchNormalization(name='tr_BatchNorm_{}'.format(blockNum), axis=-1)(previousLayer)
    relu = Activation('relu', name='tr_relu_{}'.format(blockNum))(bn)

    if dropRate is not None:
        conv = Conv2D(nOutChannels, 1, 1, border_mode='same')(relu)
        conv = BatchNormalization(axis=-1)(conv)

        conv = Activation('relu')(conv)

        avgPool = AveragePooling2D(pool_size=(2, 2))(conv)

        return avgPool
    else:
        conv = Conv2D(nOutChannels, 1, 1, border_mode='same', name='tr_conv_{}'.format(blockNum))(relu)

        conv = BatchNormalization(axis=-1)(conv)

        conv = Activation('relu')(conv)

        return conv

"""
Lambda 层是 Keras 提供的函数式 API 中的一部分，用于定义自定义的 Lambda 函数。这里的 Lambda 函数接受两个参数 x 和 repnum。
K.repeat_elements(x, repnum, axis=3) 是 Keras 的 backend（通常是 TensorFlow 或 Theano）提供的函数，用于在指定轴上重复张量元素。
这里 x 是输入张量，repnum 是重复的次数，axis=3 表示在通道轴上进行重复。
arguments={'repnum': rep} 通过 arguments 参数传递了 rep 的值给 Lambda 函数。
最终，通过 (tensor) 的方式将输入张量 tensor 传递给 Lambda 函数，从而形成一个 Keras 模型层。
"""
def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)


def expend_as_1(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1),
                  arguments={'repnum': rep})(tensor)


"""
inputs 是一个包含两个张量 m 和 n 的列表或元组。这两个张量用于计算 Jensen-Shannon 散度。
tf.clip_by_value 函数用于将输入张量中的元素限制在指定的范围内。在这里，它被用于确保 n 和 m 中的值在 (K.epsilon(), 1) 的范围内，避免取对数时出现无效值。
js 是 Jensen-Shannon 散度中的一个中间计算步骤，计算了输入概率分布的平均分布。
tf.multiply 用于逐元素相乘，这里分别计算了 js1 和 js2 中的两个部分。
tf.log 用于逐元素取对数。
tf.div 用于逐元素相除。
最后，通过加权平均 0.5 * (js1 + js2) 得到 Jensen-Shannon 散度的最终结果。
总体而言，这段代码实现了 Jensen-Shannon 散度的计算，该散度用于度量两个概率分布之间的相似性。
"""
def Jensen_Shannon_divergence(inputs):
    m, n = inputs
    n = tf.clip_by_value(n, K.epsilon(), 1)
    m = tf.clip_by_value(m, K.epsilon(), 1)

    js = (m + n) / 2

    js1 = tf.multiply(m, tf.log(tf.div(m, js)))
    js2 = tf.multiply(n, tf.log(tf.div(n, js)))
    return 0.5 * (js1 + js2)

"""
inputs 是一个包含两个张量 m 和 n 的列表或元组，这两个张量用于计算 Jensen-Shannon 散度。
Lambda(Jensen_Shannon_divergence)([m, n]) 使用了一个 Lambda 层来应用 Jensen-Shannon 散度的计算。这部分的实现可能包含了前面提到的 Jensen-Shannon 散度的计算逻辑。
Reshape((-1, 16, C))(z_1) 将 z_1 进行了形状重塑，将其变成四维张量，其中 16 和 C 是硬编码的值。
Lambda(lambda z: shape_z_1[1] - tf.reduce_sum(z, 1, keep_dims=True))(z_1) 计算了一个向量 v，其值为 shape_z_1[1] 减去 z_1 沿第一个维度的求和。
Dense(C, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(v) 应用了一个全连接层，输出维度为 C，激活函数为 sigmoid，并且没有使用偏置。
Reshape((-1, 16, C))(m) 将输入张量 m 进行了形状重塑，使其与前面计算的 f 具有相同的形状。
expend_as_1(f, shape_x_1[1]) 通过调用 expend_as_1 函数，将 f 进行扩展，使其与 x_1 具有相同的形状。
multiply([f, x_1]) 通过逐元素相乘得到 y_1。
Reshape((shape_x[1], shape_x[2], shape_x[3]))(y_1) 将结果 y_1 进行形状重塑，使其与输入张量 m 具有相同的形状。
最终，这个函数实现了基于 Jensen-Shannon 散度的重新加权，其中通过计算、调整权重，将输入张量 m 进行了加权得到输出张量 y。
"""
def Re_weight(inputs):
    m, n = inputs
    C = 16
    shape_x = K.int_shape(m)
    z_1 = Lambda(Jensen_Shannon_divergence)([m, n])

    # 16
    z_1 = Reshape((-1, 16, C))(z_1)
    shape_z_1 = K.int_shape(z_1)
    v = Lambda(lambda z: shape_z_1[1] - tf.reduce_sum(z, 1, keep_dims=True))(z_1)
    f = Dense(C, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(v)
    x_1 = Reshape((-1, 16, C))(m)
    shape_x_1 = K.int_shape(x_1)
    f = expend_as_1(f, shape_x_1[1])
    y_1 = multiply([f, x_1])
    y = Reshape((shape_x[1], shape_x[2], shape_x[3]))(y_1)
    return y


def load_data():
    mydata = dataProcess(224, 224)
    # 自定义读取数据方式
    imgs_train, imgs_mask_train = mydata.load_train_data()
    return imgs_train, imgs_mask_train


def create_model():
    # 其中shape=[224, 224, 1]指定了输入的形状。具体来说，这个模型期望接受的输入是一个三维的张量，其大小为224x224，其中的通道数为1。
    inputs = Input(shape=[280, 320, 1])

    """
    dropRate = 0.5: 这是模型中使用的丢弃率（dropout rate），表示在训练过程中随机丢弃神经元的比例，以防止过拟合。
    growthRate = [128, 128, 128, 128, 128]: 这是一个列表，包含了模型中每个密集块（dense block）的增长率。增长率是指每个密集块中每个层级产生的特征图的数量。
    nChannels = 128: 这是模型中初始的通道数，用于第一个卷积层的输出通道数。
    C = 128: 这是模型中密集块内部卷积层的输出通道数。
    N = [3, 3, 3, 3, 3]: 这是一个列表，包含了模型中每个密集块中的密集层的数量。密集层是指通过堆叠多个卷积层来构建的层级。
    在每个密集块中，都有N[i]个这样的密集层。这个列表的长度也决定了模型中密集块的数量。
    """
    dropRate = 0.5
    growthRate = [128, 128, 128, 128, 128]
    nChannels = 128
    C = 128
    N = [3, 3, 3, 3, 3]

    # encoder - 1
    # 这是一个3x3的卷积层，用于对输入图像进行特征提取。nChannels表示输出通道数，padding='same'表示使用零填充，
    # kernel_initializer='he_normal'表示使用He正态分布初始化权重。
    conv1 = Conv2D(nChannels, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    # 在卷积后进行批归一化，有助于加速训练过程并提高模型的稳定性。
    conv1 = BatchNormalization(axis=-1)(conv1)
    # 对卷积输出进行ReLU激活函数处理，引入非线性特性。
    conv1 = Activation('relu')(conv1)

    #  将卷积层的输出作为第一个密集块（dense block）的输入。
    dense_1 = conv1
    for i in range(N[0]):
        # 调用 addLayer 函数，将 dense_1 作为输入，以及密集块内部的输出通道数 C，返回密集块的输出。
        dense_1 = addLayer(dense_1, C)
        # nChannels += int(growthRate[0]): 增加通道数，用于下一个密集层。
        nChannels += int(growthRate[0])

    #  对密集块的输出进行批归一化。
    trans_1 = BatchNormalization(axis=-1)(dense_1)
    # 对批归一化后的输出进行ReLU激活。
    trans_1 = Activation('relu')(trans_1)
    # 使用1x1卷积对密集块的输出进行进一步的特征提取。
    dense_out_1 = Conv2D(C, (1, 1), padding="same", kernel_initializer='he_normal')(trans_1)
    #  调用 addTransition 函数，返回一个过渡层的输出。如果 dropRate 不为 None，则在过渡层中应用平均池化。
    trans_1 = addTransition(dense_1, C, dropRate, 1)

    # encoder - 2
    dense_2 = trans_1
    dense_2 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_2)  # nChannels = 128、C=128

    dense_2 = BatchNormalization(axis=-1)(dense_2)
    dense_2 = Activation('relu')(dense_2)
    # conv->BN->Relu
    for i in range(N[1]):
        dense_2 = addLayer(dense_2, C)
        nChannels += growthRate[1]
    dense_out_2 = BatchNormalization(axis=-1)(dense_2)
    dense_out_2 = Activation('relu')(dense_out_2)
    dense_out_2 = Conv2D(C, (1, 1), padding="same", kernel_initializer='he_normal')(dense_out_2)

    trans_2 = addTransition(dense_2, C, dropRate, 2)

    # encoder - 3
    dense_3 = trans_2
    dense_3 = BatchNormalization(axis=-1)(dense_3)
    dense_3 = Activation('relu')(dense_3)
    dense_3 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_3)
    for i in range(N[2]):
        dense_3 = addLayer(dense_3, C)
        nChannels += growthRate[2]

    trans_3 = addTransition(dense_3, C, dropRate, 3)

    dense_out_3 = BatchNormalization(axis=-1)(dense_3)
    dense_out_3 = Activation('relu')(dense_out_3)
    dense_out_3 = Conv2D(C, (1, 1), padding='same', kernel_initializer='he_normal')(dense_out_3)

    # encoder - 4
    dense_4 = trans_3
    dense_4 = BatchNormalization(axis=-1)(dense_4)
    dense_4 = Activation('relu')(dense_4)
    dense_4 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_4)

    for i in range(N[3]):
        dense_4 = addLayer(dense_4, C)
        nChannels += growthRate[3]

    trans_4 = addTransition(dense_4, C, dropRate, 4)

    dense_out_4 = BatchNormalization(axis=-1)(dense_4)
    dense_out_4 = Activation('relu')(dense_out_4)
    dense_out_4 = Conv2D(C, (1, 1), padding='same', kernel_initializer='he_normal')(dense_out_4)

    # encoder - 5
    dense_5 = trans_4
    dense_5 = BatchNormalization(axis=-1)(dense_5)
    dense_5 = Activation('relu')(dense_5)
    dense_5 = Conv2D(C, (3, 3), padding='same', kernel_initializer='he_normal')(dense_5)
    for i in range(N[4]):
        dense_5 = addLayer(dense_5, C)
        nChannels += growthRate[4]
    trans5 = addTransition(dense_5, C, None, 5)
    dense_5 = BatchNormalization(axis=-1)(dense_5)
    dense_5 = Activation('relu')(dense_5)

    dense_out_5 = Conv2D(C, (1, 1), padding='same', kernel_initializer='he_normal')(dense_5)
    dense_out_5 = BatchNormalization(axis=-1)(dense_out_5)
    dense_out_5 = Activation('relu')(dense_out_5)
    dense_out_5 = AtrousConvolution2D(C, 3, 3, atrous_rate=(2, 2))(dense_out_5)
    dense_out_5 = ZeroPadding2D(padding=(2, 2))(dense_out_5)

    dense_out_5 = BatchNormalization(axis=-1)(dense_out_5)
    dense_out_5 = Activation('relu')(dense_out_5)
    # 使用空洞卷积（Atrous Convolution）进行特征提取。Atrous卷积是在卷积核中引入空洞，有助于增大感受野，捕捉更大范围的上下文信息。
    dense_out_5 = AtrousConvolution2D(C, 3, 3, atrous_rate=(2, 2))(dense_out_5)
    # : 在进行上采样之前，对输出进行零填充，可能是为了防止图像边缘信息的丢失。
    dense_out_5 = ZeroPadding2D(padding=(2, 2))(dense_out_5)

    dense_out_1_s = dense_out_1
    #  对 "encoder - 2" 的输出进行上采样，以便与后续的层进行连接。上采样的大小是 (2, 2)。
    #  上采样的作用是增加图像或特征图的尺寸，从而提高其分辨率。在卷积神经网络中，上采样通常用于还原图像的细节，从较低分辨率的特征图生成高分辨率的输出。这在解码或反卷积部分特别有用。
    dense_out_2_s = UpSampling2D(size=(2, 2))(dense_out_2)
    dense_out_3_s = UpSampling2D(size=(4, 4))(dense_out_3)
    dense_out_4_s = UpSampling2D(size=(8, 8))(dense_out_4)
    # 进行上采样后，张量尺寸变了
    dense_out_5_s = UpSampling2D(size=(16, 16))(dense_out_5)

    # reference layer
    # 将dense_out_1_s、dense_out_2_s、dense_out_3_s、dense_out_4_s和dense_out_5_s这五个特征图相加，创建了一个新的特征图 ks。这个特征图将包含来自不同层次的信息。
    ks = Add()([dense_out_1_s, dense_out_2_s, dense_out_3_s, dense_out_4_s, dense_out_5_s])

    ks = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(ks)
    shape_K = K.int_shape(dense_out_1_s)
    sk = expend_as(ks, shape_K[3])

    # decoder
    dense_out_5 = Add()([dense_out_5, trans5])
    dense_out_5 = Dropout(0.5)(dense_out_5)
    up6 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(dense_out_5)
    merge6 = Add()([dense_out_4, up6])

    merge6 = Dropout(0.5)(merge6)
    conv6 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(conv6)
    merge7 = Add()([dense_out_3, up7])
    merge7 = Dropout(0.5)(merge7)

    conv7 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(conv7)
    merge8 = Add()([dense_out_2, up8])
    merge8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(C, (3, 3), strides=(2, 2), padding='same')(conv8)
    t_1 = Add()([dense_out_1, up9])
    t_1 = Dropout(0.5)(t_1)
    conv9 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(t_1)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(C, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Activation('relu')(conv9)

    op_1 = conv9
    op_2 = UpSampling2D(size=(2, 2))(conv8)
    op_3 = UpSampling2D(size=(4, 4))(conv7)
    op_4 = UpSampling2D(size=(8, 8))(conv6)
    op_5 = UpSampling2D(size=(16, 16))(dense_out_5)

    op_1 = Re_weight([op_1, sk])
    op_2 = Re_weight([op_2, sk])
    op_3 = Re_weight([op_3, sk])
    op_4 = Re_weight([op_4, sk])
    op_5 = Re_weight([op_5, sk])

    op = Add()([op_1, op_2, op_3, op_4, op_5])
    conv10 = Conv2D(1, 1, activation='sigmoid')(op)
    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    return model


def train():
    model_path = "Model/CAV/gaussianNoise/"

    print("got model")
    model = create_model()
    print("loading data")
    imgs_train, imgs_mask_train = load_data()
    print("loading data done")

    # 使用学习率（lr）为0.0001创建了Adam优化器的实例。Adam是用于训练神经网络的常用优化算法。
    opt = optimizers.Adam(lr=0.0001)
    # optimizer：训练期间使用的优化器。在这种情况下，是带有指定学习率的Adam优化器。
    # loss：训练期间要最小化的损失函数。"binary_crossentropy"是用于二元分类问题的常见损失函数。
    # metrics：在训练期间监视的一组指标。在这种情况下，包括"accuracy"，这是用于分类问题的常用指标。
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    """
    这里创建了一个 ModelCheckpoint 回调函数。该回调函数会在每个训练周期结束时检查模型的性能，如果模型的训练损失（monitor='loss'）有改善，
    就会保存模型权重到指定路径（model_path + 'model_new.hdf5'）。其他参数包括：
    verbose=1：在保存模型时输出信息。
    save_best_only=True：只保存在验证集上性能最好的模型。
    save_weights_only=False：保存整个模型（包括结构和权重），而不仅仅是权重。
    mode='auto'：模型检查点的保存模式，根据监测的量自动选择。
    period=1：每多少个训练周期保存一次模型。
    """
    # model_checkpoint = ModelCheckpoint(model_path + 'model_new.hdf5', monitor='loss', verbose=1,
    #                                    save_best_only=True, save_weights_only=False, mode='auto', period=1)

    model_checkpoint = ModelCheckpoint(model_path + 'kaix.hdf5', monitor='loss', verbose=1,
                                       save_best_only=True, save_weights_only=False, mode='auto', period=1)
    print('Fitting model...')
    """
    创建了一个 EarlyStopping 回调函数，它会在模型的训练损失不再改善时停止训练。参数包括：
    monitor='loss'：监测模型的训练损失。
    patience=10：如果在连续 10 个训练周期中都没有改善，就停止训练。
    verbose=0：不输出信息。
    mode='auto'：根据监测的量自动选择。
    """
    min_lr = 0.00000001
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    """
    创建了一个 ReduceLROnPlateau 回调函数，用于在训练过程中降低学习率。参数包括：
    monitor='loss'：监测模型的训练损失。
    factor=0.5：学习率降低的因子，即每次触发时将学习率乘以 0.5。
    patience=6：如果连续 6 个训练周期内都没有改善，就触发学习率降低。
    verbose=0：不输出信息。
    mode='min'：根据监测的量的最小值来触发学习率降低。
    cooldown=0：触发降低学习率后，在此周期内不再触发。
    min_lr=0.00000001：学习率的下限。
    """
    lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6, verbose=0, mode='min', cooldown=0,
                           min_lr=min_lr)
    """
    最后，使用 fit 方法来训练模型，其中包括了前面创建的回调函数。这里的训练数据是 imgs_train 和 imgs_mask_train，使用的批大小是 2，
    训练周期是 200。同时进行验证，验证集占训练数据的 20%。shuffle=True 表示每个训练周期前打乱训练数据的顺序。
    """
    epoch_num=200
    history = model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=epoch_num, verbose=1, validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_checkpoint, lr, early_stop])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    accuracy_name = 'min_lr='+str(min_lr)+'_epoch='+str(epoch_num)+'accuracy.png'
    plt.savefig(model_path + accuracy_name)
    # plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    loss_name = 'min_lr='+str(min_lr)+'_epoch='+str(epoch_num)+'loss.png'
    plt.savefig(model_path + loss_name)
    # plt.show()


if __name__ == '__main__':
    train()
    K.clear_session()
