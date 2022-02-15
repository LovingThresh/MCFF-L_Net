# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 15:07
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : TensorRT.py
# @Software: PyCharm

# 明确任务:实现tensorflow向onnx的转换
# While these two lines of command lines may seem simple, it took a week to find the right conversion
# 这个命令是成功的
#  python -m tf2onnx.convert --saved-model SaveModel --output ONNX --opset 13

# 接下来就是尝试把onnx变成trt
# trtexec --onnx=ONNX --saveEngine=saveEngine  --explicitBatch=1 --workspace=1024
