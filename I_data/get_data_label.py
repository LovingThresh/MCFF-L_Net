# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 14:54
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : get_data_label.py
# @Software: PyCharm
import cv2
import random
import numpy as np
import tensorflow as tf

train_teacher_y_name = ''
Aug_image = object
Aug_label = object
teacher_label = object
seed = int


def get_data(path=r'I:\Image Processing\train.txt',
             training=True, shuffle=True):
    """
    获取样本和标签对应的行：获取训练集和验证集的数量
    :return: lines： 样本和标签的对应行： [num_train, num_val] 训练集和验证集数量
    """

    # 读取训练样本和样本对应关系的文件 lines -> [1.jpg;1.jpg\n', '10.jpg;10.png\n', ......]
    # .jpg:样本  ：  .jpg：标签

    with open(path, 'r') as f:
        lines = f.readlines()

    # 打乱行， 打乱数据有利于训练
    if shuffle:
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

    if training:
        # 切分训练样本， 90% 训练： 10% 验证
        num_val = int(len(lines) * 0.1)
        num_train = len(lines) - num_val
        return lines, num_train, num_val

    if not training:
        num_test = len(lines)
        return lines, num_test


def get_dataset_label(lines, batch_size,
                      A_img_paths=r'I:\Image Processing\Rebuild_Image_95/',
                      B_img_paths=r'I:\Image Processing\Mix_img\95\label/',
                      shuffle=True, KD=False, training=False, Augmentation=False):
    """
        生成器， 读取图片， 并对图片进行处理， 生成（样本，标签）
        :param Augmentation:
        :param training:
        :param C_img_paths:
        :param KD:
        :param shuffle:
        :param B_img_paths:
        :param A_img_paths:
        :param lines: 样本和标签的对应行
        :param batch_size: 一次处理的图片数
        :return:  返回（样本， 标签）
        """

    global train_teacher_y_name, seed
    numbers = len(lines)
    read_line = 0
    if shuffle:
        np.random.shuffle(lines)

    while True:

        x_train = []
        y_train = []

        for t in range(batch_size):

            # 1.Get the filename
            train_x_name = lines[read_line].split(',')[0]

            # 2.Read the image
            img = cv2.imread(A_img_paths + train_x_name)
            img_array = np.array(img)
            size = (img_array.shape[0], img_array.shape[1])
            img_array = cv2.resize(img_array, size)
            img_array = img_array / 255.0  # 标准化
            img_array = img_array * 2 - 1
            x_train.append(img_array)

            train_y_name = lines[read_line].split(',')[1].replace('\n', '')
            img_array = cv2.imread(B_img_paths + train_y_name)
            img_array = cv2.resize(img_array, size)
            img_array = cv2.dilate(img_array, kernel=(3, 3), iterations=3)
            labels = np.zeros((img_array.shape[0], img_array.shape[1], 2), np.int)

            # 3.Image channels separation
            labels[:, :, 0] = (img_array[:, :, 1] == 255).astype(int).reshape(size)
            labels[:, :, 1] = (img_array[:, :, 1] != 255).astype(int).reshape(size)
            labels = labels.astype(np.float32)

            y_train.append(labels)

            # 4.Iterate over all data
            read_line = (read_line + 1) % numbers

        # 5.Data Augmentation
        image, label = np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
        if training:

            if Augmentation:
                seed = random.choice([0, 0, 0, 1, 2, 3, 4, 5, 6])
            if not Augmentation:
                seed = random.choice([0, 0, 0])

            def DataAugmentation(row_image, row_label, D_seed=0):

                global Aug_image, Aug_label
                in_seed = np.random.randint(0, 6)

                if D_seed == 0:
                    Aug_image = row_image
                    Aug_label = row_label

                if D_seed == 1:
                    Aug_image = tf.image.random_flip_left_right(row_image, seed=in_seed)
                    Aug_label = tf.image.random_flip_left_right(row_label, seed=in_seed)
                    Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                    Aug_label = np.array(np.reshape(Aug_label, row_label.shape))

                if D_seed == 2:
                    Aug_image = tf.image.random_flip_up_down(row_image, seed=in_seed)
                    Aug_label = tf.image.random_flip_up_down(row_label, seed=in_seed)
                    Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                    Aug_label = np.array(np.reshape(Aug_label, row_label.shape))

                if D_seed == 3:
                    Aug_image = tf.image.random_saturation(row_image, 0.2, 0.8)
                    Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                    Aug_label = row_label

                if D_seed == 4:
                    Aug_image = tf.image.random_contrast(row_image, 0.2, 0.8)
                    Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                    Aug_label = row_label

                if D_seed == 5:
                    Aug_image = tf.image.random_brightness(row_image, 0.5)
                    Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                    Aug_label = row_label

                if D_seed == 6:
                    Aug_image = tf.image.random_hue(row_image, 0.5)
                    Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                    Aug_label = row_label

                return Aug_image, Aug_label

            image, label = DataAugmentation(image, label, D_seed=seed)

            data = image, [label, label, label, label]

            yield data

        else:
            data = image, [label, label, label, label]

            yield data


def get_teacher_dataset_label(
                lines,
                A_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\img/',
                B_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\mask/',
                h_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_h\label/',
                x_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_x\label/',
                y_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_y\label/',
                mix_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_mix\label/',
                batch_size=1,
                shuffle=True,
                temperature=0
        ):
    numbers = len(lines)
    read_line = 0

    while True:

        y_5_train = []
        x_train = []

        for t in range(batch_size):
            y_train = []
            if shuffle:
                np.random.shuffle(lines)

            # 1.Get the filename
            train_x_name = lines[read_line].split(',')[0]

            # 2.Read the image and label
            img = cv2.imread(A_img_paths + train_x_name)
            size = (img.shape[0], img.shape[1])

            img_array = img / 255.0
            img_array = img_array * 2 - 1
            x_train.append(img_array)

            train_y_name = lines[read_line].split(',')[1].replace('\n', '')
            img_array = cv2.imread(B_img_paths + train_y_name)
            img_array = cv2.dilate(img_array, kernel=(3, 3), iterations=3)

            # 3.Image channels separation
            labels = np.zeros((img_array.shape[0], img_array.shape[1], 2), np.int)
            labels[:, :, 0] = (img_array[:, :, 1] == 255).astype(int).reshape(size)
            labels[:, :, 1] = (img_array[:, :, 1] != 255).astype(int).reshape(size)
            real_label = labels.astype(np.float32)
            if temperature > 0:
                real_label = tf.nn.softmax(real_label / temperature)
                real_label = np.array(real_label, dtype=np.float32)

            # 4.Get distillation data
            def get_label(img_paths):

                T_label = cv2.imread(img_paths + train_x_name[:-4] + '.png')
                T_label = T_label[:, :, 0:1]
                T_label = T_label / 255.0

                label_T = 1 - T_label
                T_label = np.concatenate([T_label, label_T], axis=-1)

                T_label = np.array(T_label)
                return T_label

            h_label = get_label(h_img_paths)
            x_label = get_label(x_img_paths)
            y_label = get_label(y_img_paths)
            mix_label = get_label(mix_img_paths)

            y_train.append(h_label)
            y_train.append(x_label)
            y_train.append(y_label)
            y_train.append(mix_label)

            y_train.append(real_label)
            y_5_train.append(np.array(y_train))
            read_line = (read_line + 1) % numbers

        yield np.array(x_train), [i for i in np.array(y_5_train).transpose([1, 0, 2, 3, 4])]
