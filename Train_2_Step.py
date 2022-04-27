# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 15:13
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Train.py
# @Software: PyCharm

import os
import datetime

from Metrics import *
from I_data.get_data_label import *
from Callback import *
from Model import Teacher_model, Student_model
import tensorflow.keras as keras

# Load the Teacher Model
# change the Temperature in softmax layer of four outputs and generate the distillation label
# You can get the distillation dataset from Readme.md

teacher_model_path = r''
model = keras.models.load_model(teacher_model_path,
                                custom_objects={'M_Precision': M_Precision,
                                                'M_Recall': M_Recall,
                                                'M_F1': M_F1,
                                                'M_IOU': M_IOU,
                                                'A_Precision': A_Precision,
                                                'A_Recall': A_Recall,
                                                'A_F1': A_F1,
                                                # 'mean_iou_keras': mean_iou_keras,
                                                'A_IOU': A_IOU,
                                                # 'H_KD_Loss': H_KD_Loss,
                                                # 'S_KD_Loss': S_KD_Loss,
                                                'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss,
                                                # 'DilatedConv2D': Layer.DilatedConv2D,
                                                }
                                )

# You can refer to the following procedure
# Please note the number of corresponding network layers

# --------------------------Get the distillation label---------------------------------------------

# def teacher_model(Encoder, Temperature):
#     input_layer = Encoder.input
#     h = Encoder.layers[-4].input
#     print(h.name)
#
#     x = Encoder.layers[-3].input
#     print(x.name)
#
#     y = Encoder.layers[-2].input
#     print(y.name)
#
#     mix = Encoder.layers[-1].input
#     print(mix.name)
#
#     h = h / Temperature
#     x = x / Temperature
#     y = y / Temperature
#     mix = mix / Temperature
#
#     h = keras.layers.Softmax(name='Label_h_with_Temperature')(h)
#     x = keras.layers.Softmax(name='Label_x_with_Temperature')(x)
#     y = keras.layers.Softmax(name='Label_y_with_Temperature')(y)
#     mix = keras.layers.Softmax(name='Label_mix_with_Temperature')(mix)
#
#     Teacher_model = keras.models.Model(inputs=input_layer, outputs=[h, x, y, mix])
#     Teacher_model.trainable = False
#
#     return Teacher_model
#
#
# model = teacher_model(model, temperature)
# print(model.trainable)
# if model.trainable:
#     model.trainable = False
#
# img_path: str = r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/'
#
# img_name_list = os.listdir(img_path)
# for img in img_name_list:
#     path = img_path + img
#
#     tensor = cv2.imread(path)
#     tensor = tensor / 255.0
#     tensor = tensor * 2 - 1
#     tensor = np.reshape(tensor, (1, tensor.shape[0], tensor.shape[1], tensor.shape[2]))
#     predict = model.predict(tensor)
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_h\label/'
#                + img[:-4] + '.png', np.repeat(predict[0][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_x\label/'
#                + img[:-4] + '.png', np.repeat(predict[1][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_y\label/'
#                + img[:-4] + '.png', np.repeat(predict[2][0, :, :, 0:1], 3, axis=-1))
#     plt.imsave(r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_mix\label/'
#                + img[:-4] + '.png', np.repeat(predict[3][0, :, :, 0:1], 3, axis=-1))

# ----------------------------------------------------------------------------------
#                            Step 3 Train Student Model
# ----------------------------------------------------------------------------------
