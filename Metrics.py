# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 15:02
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Metrics.py
# @Software: PyCharm

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


# KD损失函数-alpha=0.9
def S_KD_Loss(y_true, y_pred, alpha=0.9):
    soft_label_loss = Asymmetry_Binary_Loss(y_true, y_pred)

    return alpha * soft_label_loss


def H_KD_Loss(y_true, y_pred, alpha=0.9):
    hard_label_loss = Asymmetry_Binary_Loss(y_true, y_pred)

    return (1 - alpha) * hard_label_loss


def M_Precision(y_true, y_pred):
    """精确率"""

    y_pred = tf.cast(y_pred > tf.constant(0.4), tf.float32)

    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    # true positives
    tp = K.sum(K.round(K.round(K.clip(y_pred[:, :, :, 0], 0, 1)) * K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1))))
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # predicted positives
    precision = tp / (pp + 1e-8)
    return precision


# 只看核心区域
def M_Recall(y_true, y_pred):
    """召回率"""

    y_pred = tf.cast(y_pred > tf.constant(0.4), tf.float32)
    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)))  # possible positives

    recall = tp / (pp + 1e-8)
    return recall


def M_F1(y_true, y_pred):
    """F1-score"""
    precision = M_Precision(y_true, y_pred)
    recall = M_Recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def M_IOU(y_true: tf.Tensor,
          y_pred: tf.Tensor):
    y_pred = tf.cast(y_pred > tf.constant(0.4), tf.float32)
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(
        K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1)) * predict) + \
        (K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1))) - K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) *
                                                                    K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))) + \
        (K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1))) - K.sum(K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1)) *
                                                                  K.round(K.clip(y_pred[:, :, :, 0], 0, 1))))
    iou = Intersection / (Union + 1e-8)

    return iou


def A_Precision(y_true, y_pred):
    """精确率"""

    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # predicted positives
    precision = tp / (pp + 1e-8)
    return precision


def A_Recall(y_true, y_pred):
    """召回率"""

    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)))  # possible positives

    recall = tp / (pp + 1e-8)
    return recall


def A_F1(y_true, y_pred):
    """F1-score"""
    precision = A_Precision(y_true, y_pred)
    recall = A_Recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def A_IOU(y_true: tf.Tensor,
          y_pred: tf.Tensor):
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) + predict)
    iou = Intersection / (Union - Intersection + 1e-8)
    return iou


def mean_iou_keras(y_true, y_pred):
    """
    Return the mean Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the mean IoU
    """
    y_true = y_true[-1:, :, :, 0]
    y_pred = y_pred[:, :, :, 0]
    y_pred = tf.cast(y_pred > tf.constant(0.4), tf.float32)
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())

    mean_iou = 0.

    thre_list = list(np.arange(0.0000001, 0.99, 0.05))

    for thre in thre_list:
        y_pred_temp = K.cast(y_pred >= thre, K.floatx())
        y_pred_temp = K.cast(K.equal(y_pred_temp, label), K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred_temp)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred_temp) - intersection
        iou = K.switch(K.equal(union, 0), 1.0, intersection / union)
        mean_iou = mean_iou + iou

    return mean_iou / len(thre_list)


# 自定义损失函数
def Asymmetry_Binary_Loss(y_true, y_pred, alpha=200):
    # 想要损失函数更加关心裂缝的标签值1
    y_true_0, y_pred_0 = y_true[:, :, :, 0] * alpha, y_pred[:, :, :, 0] * alpha
    # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    y_true_1, y_pred_1 = y_true[:, :, :, 1], y_pred[:, :, :, 1]
    mse = tf.losses.mean_squared_error
    return mse(y_true_0, y_pred_0) + mse(y_true_1, y_pred_1) \
        + mse(y_pred[:, :, :, 0] + y_pred[:, :, :, 1], tf.ones_like(y_true[:, :, :, 0]))
