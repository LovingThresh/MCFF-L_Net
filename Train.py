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
from Model import Teacher, Student_model
import tensorflow.keras as keras

# ----------------------------------------------------------------------------------
#                              Step 1 Teacher
# ----------------------------------------------------------------------------------
train_lines, num_train = get_data(path=r'L:\ALASegmentationNets_v2\Data\Stage_4\train.txt', training=False)
validation_lines, num_val = get_data(path=r'L:\ALASegmentationNets_v2\Data\Stage_4\val.txt', training=False)
test_lines, num_test = get_data(path=r'L:\ALASegmentationNets_v2\Data\Stage_4\test.txt', training=False)
batch_size = 1
epoch = 100
train_dataset = get_dataset_label(train_lines, batch_size,
                                  A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/',
                                  B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\mask/',
                                  C_img_paths=r'C:\Users\liuye\Desktop\data\train_1\teacher_mask/',
                                  shuffle=True,
                                  KD=False,
                                  training=True,
                                  Augmentation=True)
validation_dataset = get_dataset_label(validation_lines, batch_size,
                                       A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\img/',
                                       B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\mask/',
                                       C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
                                       shuffle=False,
                                       KD=False,
                                       training=False,
                                       Augmentation=False)
test_dataset = get_dataset_label(test_lines, batch_size,
                                 A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\img/',
                                 B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\mask/',
                                 C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
                                 shuffle=False,
                                 KD=False,
                                 training=False,
                                 Augmentation=False)

model = Teacher.Teacher_model()
# Please Try different initial_learning_rate for different Dataset
# [5e-5, 2e-5, 1e-5, 5e-6, 3e-6, 1e-6]
initial_learning_rate = 3e-6

optimizer = keras.optimizers.RMSprop(initial_learning_rate)

a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
c = ''.join(b)
os.makedirs(r'E:/output/{}'.format(c))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='E:/output/{}/tensorboard/'.format(c))
checkpoint = tf.keras.callbacks.ModelCheckpoint('E:/output/{}/checkpoint/'.format(c) +
                                                'ep{epoch:03d}-val_loss{val_loss:.3f}/',
                                                # 'Output_Label_loss:.3f}-val_acc{'
                                                # 'Output_Label_accuracy:.3f}/',
                                                monitor='val_accuracy', verbose=0,
                                                save_best_only=False, save_weights_only=False,
                                                mode='auto', period=1)

os.makedirs(r'E:/output/{}/plot/'.format(c))
plot_path = r'E:/output/{}/plot/'.format(c)
checkpoints_directory = r'E:/output/{}/checkpoints/'.format(c)
checkpointplot = CheckpointPlot(generator=validation_dataset, path=plot_path)

model.compile(optimizer=optimizer,
              loss=Asymmetry_Binary_Loss,
              metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU,
                       M_Precision, M_Recall, M_F1, M_IOU, mean_iou_keras])

model.fit(train_dataset,
          steps_per_epoch=max(1, num_train // batch_size),
          epochs=epoch,
          validation_data=validation_dataset,
          validation_steps=max(1, num_val // batch_size),
          initial_epoch=0,
          callbacks=[tensorboard, checkpoint, EarlyStopping, checkpointplot,
                     DynamicLearningRate])

# ----------------------------------------------------------------------------------
#                              Step 2 Student
# ----------------------------------------------------------------------------------
train_dataset = get_teacher_dataset_label(train_lines,
                                          A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/',
                                          B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\mask/',
                                          h_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_h\label/',
                                          x_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_x\label/',
                                          y_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_y\label/',
                                          mix_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\teacher_mask\teacher_label_mix\label/',
                                          batch_size=batch_size,
                                          shuffle=True,
                                          temperature=0
                                          )

validation_dataset = get_teacher_dataset_label(validation_lines,
                                               A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\img/',
                                               B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\mask/',
                                               h_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_h\label/',
                                               x_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_x\label/',
                                               y_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_y\label/',
                                               mix_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\teacher_mask\teacher_label_mix\label/',
                                               batch_size=batch_size,
                                               shuffle=False,
                                               temperature=0,

                                               )

test_dataset = get_teacher_dataset_label(test_lines,
                                         A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\img/',
                                         B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\mask/',
                                         h_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_h\label/',
                                         x_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_x\label/',
                                         y_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_y\label/',
                                         mix_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\teacher_mask\teacher_label_mix\label/',
                                         batch_size=batch_size,
                                         shuffle=False,
                                         temperature=0
                                         )
temperature = 10

model = Student_model.Student_model()

initial_learning_rate = 5e-5

optimizer = keras.optimizers.RMSprop(initial_learning_rate)

a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
c = ''.join(b)
os.makedirs(r'E:/output/{}'.format(c))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='E:/output/{}/tensorboard/'.format(c))
checkpoint = tf.keras.callbacks.ModelCheckpoint('E:/output/{}/checkpoint/'.format(c) +
                                                'ep{epoch:03d}-val_loss{val_loss:.3f}/',
                                                # 'Output_Label_loss:.3f}-val_acc{'
                                                # 'Output_Label_accuracy:.3f}/',
                                                monitor='val_accuracy', verbose=0,
                                                save_best_only=False, save_weights_only=False,
                                                mode='auto', period=1)

os.makedirs(r'E:/output/{}/plot/'.format(c))
plot_path = r'E:/output/{}/plot/'.format(c)
checkpoints_directory = r'E:/output/{}/checkpoints/'.format(c)
checkpointplot = CheckpointPlot(generator=validation_dataset, path=plot_path)

model.compile(optimizer=optimizer,
              loss=Asymmetry_Binary_Loss,
              metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU,
                       M_Precision, M_Recall, M_F1, M_IOU, mean_iou_keras])

model.fit(train_dataset,
          steps_per_epoch=max(1, num_train // batch_size),
          epochs=epoch,
          validation_data=validation_dataset,
          validation_steps=max(1, num_val // batch_size),
          initial_epoch=0,
          callbacks=[tensorboard, checkpoint, EarlyStopping, checkpointplot,
                     DynamicLearningRate])


# ----------------------------------------------------------------------------------
#                            Step 3 Deploy Student Model
# ----------------------------------------------------------------------------------
