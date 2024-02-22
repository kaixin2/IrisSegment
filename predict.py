from model import *
from keras import backend as K
from keras.preprocessing.image import array_to_img
import cv2
import glob
import numpy as np
import logging


async def predict_file(model_path='Model/CAV/gaussianNoise/model_new.hdf5', file_path=''):
    try:
        imgs = glob.glob(file_path)
        model = create_model()
        logging.warning("predict.py => model_path = " + model_path)
        model.load_weights(model_path)
        img_result_path = ''
        file_pre_path = file_path[0:file_path.rindex("/") + 1]
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
            img_result_path = file_pre_path + midname + "tiff"
            img.save(img_result_path)
        K.clear_session()
        return img_result_path
    except Exception as e:
        print(f"Error in callback: {e}")


async def batch_predict_file(model_path='Model/CAV/gaussianNoise/model_new.hdf5', file_directory='', store_path=''):
    try:
        logging.warning("predict.py => file_directory = " + file_directory)
        logging.warning("predict.py => store_path = " + store_path)
        img_type = "jpg"
        # read_pre_path = file_directory[0:file_directory.rindex("/") + 1]

        imgs = glob.glob(file_directory + "/*." + img_type)
        model = create_model()
        model.load_weights(model_path)

        for imgname in imgs:
            # 使用OpenCV (cv2) 读取一张图像 (imgname)，并将其转换为灰度图。图像的像素值被转换为浮点型。
            image_rgb = (np.array(cv2.imread(imgname, 0))).astype(np.float32)
            # 这里将灰度图转换为Keras模型的输入格式。np.expand_dims 用于在最后一个轴上增加一个维度，以适应模型的输入要求。然后，将像素值缩放到 [0, 1] 范围内，通过除以 255。
            image = np.expand_dims(image_rgb, axis=-1) / 255
            # 创建一个形状为 (1, 224, 224, 1) 的零矩阵 net_in，用于作为模型的输入。将前面处理好的图像数据 image 放入这个矩阵中。
            net_in = np.zeros((1, 224, 224, 1), dtype=np.float32)
            net_in[0] = image

            logging.warning("predict.py => imaname = " + imgname)
            # 这里从图像文件路径中提取出文件名（不包括路径和扩展名），将其存储在 midname 中。
            midname = imgname[imgname.rindex("\\") + 1:imgname.rindex(".") + 1]
            imgs_mask_test = model.predict(net_in)[0]

            img = imgs_mask_test
            # 将预测得到的图像保存为 TIFF 文件。array_to_img 将数组转换为 PIL 图像对象，然后使用 img.save 将图像保存到指定路径。
            # 文件名由之前提取的 midname 加上后缀 "tiff" 构成。这段代码的作用是将模型对输入图像的预测结果保存为图像文件。
            img = array_to_img(img)
            img_result_path = store_path + '/' + midname + "tiff"
            logging.warning("predict.py => final_path = " + img_result_path)
            logging.warning("predict.py => midname = " + midname)
            img.save(img_result_path)
        K.clear_session()
        return "success"
    except Exception as e:
        print(f"Error in callback: {e}")
