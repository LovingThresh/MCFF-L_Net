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
from Model import Student_model
import tensorflow.keras as keras
from I_data.get_data_label import *


# You Choose any batch size and epoch
batch_size = 1
epoch = 100

# ----------------------------------------------------------------------------------
#                              Step 3 Student Training
# ----------------------------------------------------------------------------------

# Please Pay Attention
# For different distillation strategies please follow the settings in the paper
# Please modify the dataset loading function accordingly
# The key parameter is the temperature
# This example is standardKD MCL

train_lines, num_train = get_data(path='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train.txt', training=False)
validation_lines, num_val = get_data(path='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val.txt', training=False)
test_lines, num_test = get_data(path='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test.txt', training=False)

train_dataset = get_teacher_dataset_label(train_lines,
                                          A_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train/img/',
                                          B_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train/mask/',
                                          h_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train/teacher_mask/teacher_label_h/label/',
                                          x_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train/teacher_mask/teacher_label_x/label/',
                                          y_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train/teacher_mask/teacher_label_y/label/',
                                          mix_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/train/teacher_mask/teacher_label_mix/label/',
                                          batch_size=batch_size,
                                          shuffle=True,
                                          temperature=0
                                          )

validation_dataset = get_teacher_dataset_label(validation_lines,
                                               A_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val/img/',
                                               B_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val/mask/',
                                               h_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val/teacher_mask/teacher_label_h/label/',
                                               x_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val/teacher_mask/teacher_label_x/label/',
                                               y_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val/teacher_mask/teacher_label_y/label/',
                                               mix_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/val/teacher_mask/teacher_label_mix/label/',
                                               batch_size=batch_size,
                                               shuffle=False,
                                               temperature=0,

                                               )

test_dataset = get_teacher_dataset_label(test_lines,
                                         A_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test/img/',
                                         B_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test/mask/',
                                         h_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test/teacher_mask/teacher_label_h/label/',
                                         x_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test/teacher_mask/teacher_label_x/label/',
                                         y_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test/teacher_mask/teacher_label_y/label/',
                                         mix_img_paths='/root/autodl-tmp/ALASegmentationNets_v2/Data/Stage_4/test/teacher_mask/teacher_label_mix/label/',
                                         batch_size=batch_size,
                                         shuffle=False,
                                         temperature=0
                                         )


temperature = 10

# You can view the student network code and modify it to MCLD
# Now it is MCL
model = Student_model.Student_model_standardKD()
profile = model_profiler.model_profiler(model, batch_size)
print(profile)
model.summary()

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

# Then get a Student model on standardKD

# -----------> Retain only one output (Label_mix_for_real)  -----------> Step 4 (TensorRT.py)
# Model Conversion and Optimization
# ----------------------------------------------------------------------------------
#                            Step 4 Deploy Student Model
#                                   TensorRT.py
# ----------------------------------------------------------------------------------
