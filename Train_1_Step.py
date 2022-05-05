# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 15:13
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Train.py
# @Software: PyCharm

import os
import datetime
import model_profiler

from Metrics import *
from Callback import *
from Model import Teacher_model
import tensorflow.keras as keras
from I_data.get_data_label import *

batch_size = 2
epoch = 100
path = 'L:'
# ----------------------------------------------------------------------------------
#                              Step 1 Teacher Training
# ----------------------------------------------------------------------------------
train_lines, num_train = get_data(path='{}/ALASegmentationNets_v2/Data/Stage_4/train.txt'.format(path), training=False)
validation_lines, num_val = get_data(path='{}/ALASegmentationNets_v2/Data/Stage_4/val.txt'.format(path),
                                     training=False)
test_lines, num_test = get_data(path='{}/ALASegmentationNets_v2/Data/Stage_4/test.txt'.format(path), training=False)

train_dataset = get_dataset_label(train_lines, batch_size,
                                  A_img_paths='{}/ALASegmentationNets_v2/Data/Stage_4/train/img/'.format(path),
                                  B_img_paths='{}/ALASegmentationNets_v2/Data/Stage_4/train/mask/'.format(path),
                                  shuffle=True,
                                  KD=False,
                                  training=True,
                                  Augmentation=True)
validation_dataset = get_dataset_label(validation_lines, batch_size,
                                       A_img_paths='{}/ALASegmentationNets_v2/Data/Stage_4/val/img/'.format(path),
                                       B_img_paths='{}/ALASegmentationNets_v2/Data/Stage_4/val/mask/'.format(path),
                                       shuffle=False,
                                       KD=False,
                                       training=False,
                                       Augmentation=False)

test_dataset = get_dataset_label(test_lines, batch_size,
                                 A_img_paths='{}/ALASegmentationNets_v2/Data/Stage_4/test/img/'.format(path),
                                 B_img_paths='{}/ALASegmentationNets_v2/Data/Stage_4/test/mask/'.format(path),
                                 shuffle=False,
                                 KD=False,
                                 training=False,
                                 Augmentation=False)

# train_dataset = get_dataset_label(train_lines, batch_size,
#                                   A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/',
#                                   B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\mask/',
#                                   shuffle=True,
#                                   KD=False,
#                                   training=True,
#                                   Augmentation=True)
# validation_dataset = get_dataset_label(validation_lines, batch_size,
#                                        A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\img/',
#                                        B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\val\mask/',
#                                        shuffle=False,
#                                        KD=False,
#                                        training=False,
#                                        Augmentation=False)
# test_dataset = get_dataset_label(test_lines, batch_size,
#                                  A_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\img/',
#                                  B_img_paths=r'L:\ALASegmentationNets_v2\Data\Stage_4\test\mask/',
#                                  C_img_paths=r'C:\Users\liuye\Desktop\data\val\teacher_mask/',
#                                  shuffle=False,
#                                  KD=False,
#                                  training=False,
#                                  Augmentation=False)

model = Teacher_model.Teacher_model()
profile = model_profiler.model_profiler(model, batch_size)
print(profile)

# Please Try different initial_learning_rate for different Dataset(Stage_1, Stage_2, Stage_4)
# The model of paper is trained on Stage_4 Dataset

# learning rates are from [5e-5, 2e-5, 1e-5, 5e-6, 3e-6, 1e-6]
a = train_dataset.__next__()
initial_learning_rate = 5e-5

optimizer = keras.optimizers.RMSprop(initial_learning_rate)

a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
c = ''.join(b)
os.makedirs(r'./output/{}'.format(c))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./output/{}/tensorboard/'.format(c))
checkpoint = tf.keras.callbacks.ModelCheckpoint('./output/{}/checkpoint/'.format(c) +
                                                'ep{epoch:03d}-val_loss{val_loss:.3f}/',
                                                # 'Output_Label_loss:.3f}-val_acc{'
                                                # 'Output_Label_accuracy:.3f}/',
                                                monitor='val_accuracy', verbose=0,
                                                save_best_only=False, save_weights_only=False,
                                                mode='auto', period=1)

os.makedirs(r'./output/{}/plot/'.format(c))
plot_path = r'./output/{}/plot/'.format(c)
checkpoints_directory = r'./output/{}/checkpoints/'.format(c)
checkpointplot = CheckpointPlot(generator=validation_dataset, path=plot_path)

model.compile(optimizer=optimizer,
              loss=Asymmetry_Binary_Loss,
              metrics=['accuracy', A_Precision, A_Recall, A_F1, A_IOU,
                       M_Precision, M_Recall, M_F1, M_IOU])

model.fit(train_dataset,
          steps_per_epoch=max(1, num_train // batch_size),
          epochs=epoch,
          validation_data=validation_dataset,
          validation_steps=max(1, num_val // batch_size),
          initial_epoch=0,
          callbacks=[tensorboard, checkpoint, EarlyStopping, checkpointplot,
                     DynamicLearningRate])

# Then get a Teacher model

# -----------> Need generate distillation label (Soft Label) -----------> Step 2
