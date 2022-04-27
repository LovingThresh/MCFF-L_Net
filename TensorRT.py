# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 15:07
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : TensorRT.py
# @Software: PyCharm


# While these two lines of command lines may seem simple, it took a week to find the right conversion

#  python -m tf2onnx.convert --saved-model SaveModel --output ONNX --opset 13

#  trtexec --onnx=ONNX --saveEngine=saveEngine  --explicitBatch=1 --workspace=1024

# You can get the TensorFlow Model and ONNX Model from Readme.md

# ----------------------------------------------------------------------------------
#                            Step 4 Deploy Student Model
#                                   Application.py
# ----------------------------------------------------------------------------------
