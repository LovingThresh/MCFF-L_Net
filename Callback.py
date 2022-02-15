# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 15:00
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Callback.py
# @Software: PyCharm
import tensorflow.keras as keras
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(save_path, predict_array):
    if predict_array.ndim == 4:
        sns.heatmap(predict_array[:, :, :, 0].reshape((predict_array.shape[1], predict_array.shape[2])),
                    xticklabels=False, yticklabels=False)
    else:
        sns.heatmap(predict_array[:, :, 0].reshape((predict_array.shape[0], predict_array.shape[1])),
                    xticklabels=False, yticklabels=False)
    plt.savefig(save_path)


DynamicLearningRate = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=1e-8
)

EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')


# 2、在每个Epoch后输出X张预测图片
class CheckpointPlot(keras.callbacks.Callback):
    def __init__(self, generator, path, num_img=1):
        super(CheckpointPlot, self).__init__()
        self.generator = generator
        self.father_path = path
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.num_img):
            raw_tuples = self.generator.__next__()
            raw_image = raw_tuples[0]
            raw_label = raw_tuples[1]
            predict_array = self.model.predict(raw_image)
            save_predict_path = self.father_path + '{}_{}_Predict_'.format(str(epoch), i) + '.png'
            save_true_path = self.father_path + '{}_{}_True_'.format(str(epoch), i) + '.png'
            plt.figure()
            plot_heatmap(save_path=save_predict_path, predict_array=predict_array[-1])
            plt.figure()
            plot_heatmap(save_path=save_true_path, predict_array=raw_label[-1])
