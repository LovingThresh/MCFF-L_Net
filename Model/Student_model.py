# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 12:45
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : Student_model.py
# @Software: PyCharm


from Segmentation_with_Three_Channels import ResnetGenerator_with_ThreeChannel


# The  Student model has a structure similar to that of the teacher model
def Student_model():
    model = ResnetGenerator_with_ThreeChannel(input_shape=(448, 448, 3),
                                              output_channels=2,
                                              dim=16,
                                              n_downsamplings=2,
                                              n_blocks=4,
                                              norm='instance_norm',
                                              attention=True,
                                              ShallowConnect=False,
                                              Temperature=10,
                                              StudentNet=True,
                                              mix_for_real_Temperature=1)

    return model


model = Student_model()
model.summary()
