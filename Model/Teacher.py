# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 12:21
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Teacher.py
# @Software: PyCharm

from Segmentation_with_Three_Channels import ResnetGenerator_with_ThreeChannel


# The  Teacher model has enough parameters to guarantee the richness of information
def Teacher_model():
    model = ResnetGenerator_with_ThreeChannel(input_shape=(448, 448, 3),
                                              output_channels=2,
                                              dim=64,
                                              n_downsamplings=2,
                                              n_blocks=8,
                                              norm='instance_norm',
                                              attention=True,
                                              ShallowConnect=False,
                                              Temperature=0,
                                              StudentNet=False,
                                              mix_for_real_Temperature=1)

    return model
