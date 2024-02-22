import logging

from model import *
from util import *
from keras import backend as K
from keras.preprocessing.image import array_to_img
import cv2
import glob
import numpy as np


if __name__ == "__main__":
    data_path = "Data/CAV/guassian_noise_224/test/img/"

    model_path = "Model/CAV/gaussianNoise/"

    img_type = "jpg"

    imgs = glob.glob(data_path + "/*." + img_type)

    # import the model
    model = create_model()

    # load the model
    # 将之前保存的模型权重加载到当前的模型中
    model.load_weights(model_path + 'model_new.hdf5')

    for imgname in imgs:
        # 使用OpenCV (cv2) 读取一张图像 (imgname)，并将其转换为灰度图。图像的像素值被转换为浮点型。
        image_rgb = (np.array(cv2.imread(imgname, 0))).astype(np.float32)
        # 这里将灰度图转换为Keras模型的输入格式。np.expand_dims 用于在最后一个轴上增加一个维度，以适应模型的输入要求。然后，将像素值缩放到 [0, 1] 范围内，通过除以 255。
        image = np.expand_dims(image_rgb, axis=-1) / 255
        # 创建一个形状为 (1, 224, 224, 1) 的零矩阵 net_in，用于作为模型的输入。将前面处理好的图像数据 image 放入这个矩阵中。
        net_in = np.zeros((1, 224, 224, 1), dtype=np.float32)
        net_in[0] = image

        # 这里从图像文件路径中提取出文件名（不包括路径和扩展名），将其存储在 midname 中。
        midname = imgname[imgname.rindex("/") + 1:imgname.rindex(".") + 1]

        imgs_mask_test = model.predict(net_in)[0]

        img = imgs_mask_test
        # 将预测得到的图像保存为 TIFF 文件。array_to_img 将数组转换为 PIL 图像对象，然后使用 img.save 将图像保存到指定路径。
        # 文件名由之前提取的 midname 加上后缀 "tiff" 构成。这段代码的作用是将模型对输入图像的预测结果保存为图像文件。
        img = array_to_img(img)
        img.save(model_path + midname + "tiff")

    """
    这行代码使用了 Keras 的 clear_session 函数。让我解释一下这个函数的作用：
    在使用 Keras 构建和训练深度学习模型时，会创建一个计算图（computation graph），其中包含了模型的各种层、连接等信息。
    clear_session 函数的作用是清除当前的计算图，释放内存资源，并重置 Keras 的全局状态。这对于释放 GPU 内存以及在一个脚本中多次构建和训练模型时是很有用的
    """
    K.clear_session()
