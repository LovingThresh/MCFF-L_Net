# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 14:32
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : camera.py
# @Software: PyCharm
import numpy as np  # 引入numpy 用于矩阵运算
import cv2  # 引入opencv库函数
import tensorrt as trt
import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


# 创建一个video capture的实例
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# 画面高度度设定为 1080
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

cv2.namedWindow('image_win', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# 图像计数 从1开始
img_count = 1


# 载入TensorRT 引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


# !!Your TensorRT file path!!
# engine_file_path = r'I:\Image Processing\ep574_32_4_1024engine.trt'


def load_normalized_data(data, pagelocked_buffer, target_size=(448, 448)):
    # upsample_size = [int(target_size[1] / 8 * 4.0), int(target_size[0] / 8 * 4.0)]
    img = data
    img = img / 255.
    img = img * 2 - 1
    # 此时img.shape为H * W * C: 432, 848, 3
    # print("图片shape", img.shape)
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    # np.copyto(pagelocked_buffer, img.ravel())
    np.copyto(pagelocked_buffer, img.ravel())
    return img


engine = load_engine(trt_runtime, engine_file_path)


while True:
    # 逐帧获取画面
    # 如果画面读取成功 ret=True，frame是读取到的图片对象(numpy的ndarray格式)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (448, 448))
    if not ret:
        # 如果图片没有读取成功
        print("图像获取失败，请按照说明进行问题排查")
        # 读取失败？问题排查
        # **驱动问题** 有的摄像头可能存在驱动问题，需要安装相关驱动，或者查看摄像头是否有UVC免驱协议
        # **接口兼容性问题** 或者USB2.0接口接了一个USB3.0的摄像头，也是不支持的。
        # **设备挂载问题** 摄像头没有被挂载，如果是虚拟机需要手动勾选设备
        # **硬件问题** 在就是检查一下USB线跟电脑USB接口
        break

    # 颜色空间变换
    # 将BGR彩图变换为灰度图
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 图片镜像
    # * 水平翻转 flipCode = 1
    # * 垂直翻转 flipCode = 0
    # * 同时水平翻转与垂直翻转 flipCode = -1
    #
    # flipCode = -1
    # frame = cv2.flip(frame, flipCode)

    # 更新窗口“image_win”中的图片
    cv2.imshow('image_win', frame)

    input_image_path = frame
    input_image_path = cv2.resize(input_image_path, (448, 448))
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        upsample = load_normalized_data(input_image_path, pagelocked_buffer=inputs[0].host)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    a = trt_outputs[0].reshape((448, 448, 2))
    y_pred = np.uint8(np.around(a[:, :, 0])).reshape(448, 448, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    frame_x = frame * (1 - y_pred) + y_pred * 255

    cv2.imshow('image_result', frame_x)

    # 等待按键事件发生 等待1ms
    key = cv2.waitKey(1)
    if key == ord('q'):
        # 如果按键为q 代表quit 退出程序
        break
    elif key == ord('c'):
        # 如果c键按下，则进行图片保存
        # 写入图片 并命名图片为 图片序号.png
        cv2.imwrite("{}.png".format(img_count), frame)
        print("截图，并保存为  {}.png".format(img_count))
        # 图片编号计数自增1
        img_count += 1

# 释放VideoCapture
cap.release()
# 销毁所有的窗口
cv2.destroyAllWindows()
