# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : some_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-02 22:09:01
"""

import tensorflow as tf
import numpy as np


def softmax(x,axis):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = axis, keepdims = True)
    s = x_exp / x_sum
    return s

A = [[0.1,0.1,0.5,0.3],
     [0.2,0.2,0.5,0.1]]
axis = 1  # 默认计算最后一维

#[1]使用自定义softmax
s1=softmax(A,axis=axis)
print("s1:{}".format(s1))

#[2]使用TF的softmax
with tf.Session() as sess:
    tf_s2=tf.nn.softmax(A, axis=axis)
    s2=sess.run(tf_s2)
    print("s2:{}".format(s2))
