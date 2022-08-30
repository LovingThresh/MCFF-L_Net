# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 22:48
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : PC.py
# @Software: PyCharm

import cv2
import zmq
import base64
import numpy as np

context = zmq.Context()
footage_socket = context.socket(zmq.PAIR)
footage_socket.bind('tcp://*:5555')

while True:
    frame = footage_socket.recv()
    img = base64.b64decode(frame)
    npimg = np.frombuffer(img, dtype=np.uint8)
    source = cv2.imdecode(npimg, 1)
    cv2.imshow("stream", source)
    cv2.waitKey(1)
