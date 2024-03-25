import os
import cv2


def test1():
    file_path = "Data/CAV/guassian_noise_224/train/img\S1001L07.jpg"

    # 获取文件名
    file_name = os.path.basename(file_path)

    # 使用split方法分割文件名，以'.'为分隔符，取第一部分
    desired_part = file_name.split('.')[0] + '.'

    print(desired_part)


def resize_image(img_path):
    # 读取图片
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (224, 224))
    return resized_img

def test2():
    # 调整尺寸后的图片路径
    resized_img_path = "test/result.jpg"
    # 原始图片路径
    original_img_path = "test/S1001L01.jpg"

    # 调用方法进行图片尺寸调整
    resized_img = resize_image(original_img_path)

    # 保存调整尺寸后的图片
    cv2.imwrite(resized_img_path, resized_img)


if __name__ == '__main__':
   test2()