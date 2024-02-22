from keras.preprocessing.image import img_to_array
import numpy as np
import glob
import cv2


class dataProcess(object):
    def __init__(self, out_rows, out_cols,
                 data_path='Data/CAV/guassian_noise_224/train/img/',
                 label_path='Data/CAV/guassian_noise_224/train/mask/',
                 npy_path="Data/CAV/guassian_noise_224/", img_type="JPG"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)

        print(len(imgs))
        """
        这行代码创建了一个 NumPy 多维数组（np.ndarray），表示图像数据。具体来说，这个数组是一个四维数组，其形状是 (len(imgs), self.out_rows, self.out_cols, 1)，
        数据类型是 np.uint8。len(imgs) 是数组的第一个维度，表示数组中有多少个图像。每个图像都对应数组的一个切片。
        self.out_rows 是数组的第二个维度，表示每个图像的行数。
        self.out_cols 是数组的第三个维度，表示每个图像的列数。
        1 是数组的第四个维度，表示每个像素的通道数。这里是灰度图，因此只有一个通道。
        """
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex(".") + 1]

            img = cv2.imread(self.data_path + "/" + midname + self.img_type, 0)
            label = cv2.imread(self.label_path + "/" + midname + "tiff", 0)
            img = img_to_array(img)
            label = img_to_array(label)

            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        # np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('load train util images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        print(imgs_train.shape)
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train


if __name__ == "__main__":
    mydata = dataProcess(224, 224)
    mydata.create_train_data()
