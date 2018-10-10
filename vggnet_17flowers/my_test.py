#-*-coding:utf-8-*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : my_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-02 10:27:56
"""
import numpy as np

data_dict = np.load('./vgg16.npy', encoding='latin1').item()
print(data_dict)