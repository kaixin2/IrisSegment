import numpy as np
import cv2
import glob
from keras.preprocessing.image import img_to_array


def ac_error(y_true, y_pred):
    y_pred = np.round(y_pred / 255)
    y_true = np.round(y_true / 255)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_true[y_true > 0.5] = 1
    y_true[y_true <= 0.5] = 0
    # TP y_true=1 y_pred=1
    y_tp = y_true + y_pred
    y_tp[y_tp != 2] = 0
    y_tp[y_tp == 2] = 1
    tp = np.sum(y_tp)

    # FN y_true=1 y_pred=0
    y_fn = y_true - y_pred
    y_fn[y_fn != 1] = 0
    y_fn[y_fn == 1] = 1
    fn = np.sum(y_fn)

    # TN y_true=0 y_pred=0
    y_fn = y_true + y_pred
    y_fn[y_fn != 0] = 0
    y_fn[y_fn == 0] = 1
    tn = np.sum(y_fn)

    # FP y_true=0 y_pred=1
    y_fp = y_true - y_pred
    y_fp[y_fp != -1] = 0
    y_fp[y_fp == -1] = 1
    fp = np.sum(y_fp)

    R = tp / (tp + fn)
    P = tp / (tp + fp)

    F = 2 * tp / (2 * tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fn + fp)

    return R, P, F, accuracy

"""
这段代码计算了二进制分割任务中的几个性能指标，包括 Recall (R)，Precision (P)，F1 Score (F)，和 Accuracy。以下是对代码的分析：
ac_error(y_true, y_pred) 函数计算了 Recall、Precision、F1 Score 和 Accuracy。该函数的输入为真实标签 y_true 和预测标签 y_pred，并返回这些性能指标。
在主函数中，通过 glob 模块获取了预测和真实标签的文件路径列表 imgs_pred 和 imgs_true。
使用 np.ndarray 创建了两个数组 img_pred 和 img_true，分别用于存储预测标签和真实标签的图像数据。
使用循环遍历了每个预测标签图像，并加载相应的真实标签图像。然后，将这两个图像转换为数组，并调用 ac_error 函数计算性能指标。
打印了每个图像的性能指标，包括 F1 Score、Recall、Precision 和 Accuracy。同时，计算了这些指标的平均值和标准差。
最后，输出了整体的平均和标准差，以及每个图像的性能指标。
这段代码的目的是评估模型在二进制分割任务中的性能，并输出每个图像的性能指标，以及整体的平均和标准差。
"""
if __name__ == "__main__":
    pred_path = "Model/CAV/gaussianNoise/"

    true_path = "Data/CAV/guassian_noise_224/test/mask/"

    print('-' * 30)
    print('load predict npydata...')
    print('-' * 30)
    imgs_pred = glob.glob(pred_path + "/*.tiff")

    print('-' * 30)
    print('load ground_true npydata...')
    print('-' * 30)
    imgs_true = glob.glob(true_path + "/*.tiff")

    img_pred = np.ndarray((len(imgs_pred), 224, 224, 1), dtype=np.uint8)
    img_true = np.ndarray((len(imgs_pred), 224, 224, 1), dtype=np.uint8)

    i = 0
    R_sum = 0
    P_sum = 0
    F_sum = 0

    R_list = []
    P_list = []
    F_list = []
    A_list = []
    for imgname in imgs_pred:
        midname = imgname[imgname.rindex("/") + 1:]
        y_true = cv2.imread(true_path + midname, 0)
        y_pred = cv2.imread(imgname, 0)

        y_true = img_to_array(y_true)
        y_pred = img_to_array(y_pred)
        R, P, F, accuracy = ac_error(y_true, y_pred)
        print(30 * "_")

        R_list.append(R)
        P_list.append(P)
        F_list.append(F)
        A_list.append(accuracy)

        # 对于一个一维数组，np.mean 返回数组中所有元素的平均值；对于多维数组，可以指定 axis 参数以计算沿特定轴的平均值。
        R_sum = np.mean(R_list)
        P_sum = np.mean(P_list)
        F_sum = np.mean(F_list)
        A_sum = np.mean(A_list)

        R_std = np.sqrt(((R_list - np.mean(R_list)) ** 2).sum() / len(R_list))
        P_std = np.sqrt(((P_list - np.mean(P_list)) ** 2).sum() / len(P_list))
        F_std = np.sqrt(((F_list - np.mean(F_list)) ** 2).sum() / len(F_list))
        A_std = np.sqrt(((A_list - np.mean(A_list)) ** 2).sum() / len(A_list))

        print(i, midname, ":")
        print("F :", F_sum, F_std)
        print("R :", R_sum, R_std)
        print("P :", P_sum, P_std)
        print("Error :", 1 - A_sum, A_std)

        i += 1
