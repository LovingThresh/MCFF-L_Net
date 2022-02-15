# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 12:21
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Teacher.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

mix, f3, f2, f1, mix_for_real = object, object, object, object, object


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator_with_ThreeChannel(input_shape=(448, 448, 3),
                                      output_channels=2,
                                      dim=64,
                                      n_downsamplings=2,
                                      n_blocks=8,
                                      norm='instance_norm',
                                      attention=False,
                                      ShallowConnect=False,
                                      Temperature=0,
                                      StudentNet=False,
                                      mix_for_real_Temperature=1):
    global mix, f3, f2, f1, mix_for_real
    Norm = _get_norm_layer(norm)
    if attention:
        output_channels = output_channels + 1

    # 受保护的用法
    def _residual_block(x_res):
        x_dim = x_res.shape[-1]
        h_res = x_res

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        h_res = tf.pad(h_res, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')

        h_res = keras.layers.Conv2D(x_dim, 3, padding='valid', use_bias=False)(h_res)
        h_res = Norm()(h_res)
        h_res = tf.nn.relu(h_res)

        h_res = tf.pad(h_res, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h_res = keras.layers.Conv2D(x_dim, 3, padding='valid', use_bias=False)(h_res)
        h_res = Norm()(h_res)

        return keras.layers.add([x_res, h_res])

    # 0
    h = inputs = keras.Input(shape=input_shape)
    # h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')

    # x ---> Dilation Convolution
    x = h
    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(3, 3), use_bias=False)(x)
    x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(x)
    x = Norm()(x)
    x = tf.nn.relu(x)

    # y ---> Rectangle Convolution
    y = h
    y1 = keras.layers.Conv2D(dim, (7, 3), strides=1, padding='same', use_bias=False)(y)
    y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y1)
    y2 = keras.layers.Conv2D(dim, (3, 7), strides=1, padding='same', use_bias=False)(y)
    y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y2)
    y = keras.layers.Add()([y1, y2])
    y = keras.layers.Conv2D(dim, 7, padding='same', use_bias=False)(y)
    y = Norm()(y)
    y = tf.nn.relu(y)

    # h ---> Standard Convolution
    # h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
    h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
    h = keras.layers.Conv2D(dim, 7, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    if ShallowConnect:
        f1 = h

    # 2
    for i in range(n_downsamplings):
        dim *= 2

        x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(3, 3), use_bias=False)(x)
        x = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = Norm()(x)
        x = tf.nn.relu(x)

        y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y)
        y1 = keras.layers.Conv2D(dim, (3, 1), strides=1, padding='same', use_bias=False)(y1)
        y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y)
        y2 = keras.layers.Conv2D(dim, (1, 3), strides=1, padding='same', use_bias=False)(y2)

        y = keras.layers.Add()([y1, y2])
        y = keras.layers.MaxPooling2D((2, 2))(y)
        y = Norm()(y)
        y = tf.nn.relu(y)

        h = keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), use_bias=False)(h)
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        if (i == 0) & ShallowConnect:
            f2 = h

    if ShallowConnect:
        f3 = h
    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)
        x = _residual_block(x)
        y = _residual_block(y)

    if ShallowConnect:
        h = keras.layers.concatenate([h, f3], axis=-1)

    # 4
    for _ in range(n_downsamplings):
        if (_ == 1) & ShallowConnect:
            h = keras.layers.concatenate([h, f2], axis=-1)
        dim //= 2

        # When model is Training as teacher model, we need Dropout Layer to restrain overfit
        if not StudentNet:
            h = keras.layers.Dropout(0.5)(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        if not StudentNet:
            x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        if not StudentNet:
            y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(y)
        y = Norm()(y)
        y = tf.nn.relu(y)

    # 5
    if ShallowConnect:
        h = keras.layers.concatenate([h, f1], axis=-1)
    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')

    if input_shape == (227, 227, 3):
        h = keras.layers.Conv2D(output_channels, 8, padding='valid')(h)
    else:
        h = keras.layers.Conv2D(output_channels, 7, padding='same', use_bias=False)(h)

        x = keras.layers.Conv2D(output_channels, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(3, 3),
                                use_bias=False)(x)
        x = keras.layers.Conv2D(output_channels, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2),
                                use_bias=False)(x)

        y1 = keras.layers.Conv2D(output_channels, (7, 3), strides=1, padding='same', use_bias=False)(y)
        y1 = keras.layers.Conv2D(output_channels, (3, 1), strides=1, padding='same', use_bias=False)(y1)
        y2 = keras.layers.Conv2D(output_channels, (3, 7), strides=1, padding='same', use_bias=False)(y)
        y2 = keras.layers.Conv2D(output_channels, (1, 3), strides=1, padding='same', use_bias=False)(y2)
        y = keras.layers.Add()([y1, y2])
        y = keras.layers.Conv2D(output_channels, 7, padding='same', use_bias=False)(y)

        mix = keras.layers.Add()([h, y, x])
    if attention:
        attention_mask = tf.sigmoid(h[:, :, :, 0])
        content_mask = h[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        h = content_mask * attention_mask

        attention_mask = tf.sigmoid(x[:, :, :, 0])
        content_mask = x[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        x = content_mask * attention_mask

        attention_mask = tf.sigmoid(y[:, :, :, 0])
        content_mask = y[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        y = content_mask * attention_mask

        attention_mask = tf.sigmoid(mix[:, :, :, 0])
        content_mask = mix[:, :, :, 1:]
        attention_mask = tf.expand_dims(attention_mask, axis=3)
        attention_mask = tf.concat([attention_mask, attention_mask], axis=3)
        mix = content_mask * attention_mask
    # h = tf.tanh(h)

    if (Temperature != 0) & StudentNet:
        h = h / Temperature
        x = x / Temperature
        y = y / Temperature
        mix_for_real = mix
        mix = mix / Temperature
        mix_for_real = keras.layers.Softmax(name='Label_mix_for_real')(mix_for_real / mix_for_real_Temperature)

    h = keras.layers.Softmax(name='Label_h')(h)
    x = keras.layers.Softmax(name='Label_x')(x)
    y = keras.layers.Softmax(name='Label_y')(y)
    mix = keras.layers.Softmax(name='Label_mix')(mix)

    # For Training
    if not Temperature:
        return keras.Model(inputs=inputs, outputs=[h, x, y, mix])

    # For Knowledge Distillation
    if Temperature:
        return keras.Model(inputs=inputs, outputs=[h, x, y, mix, mix_for_real])
