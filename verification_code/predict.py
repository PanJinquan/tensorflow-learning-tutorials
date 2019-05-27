# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : predict.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-02 20:37:09
"""

# coding=utf-8

import tensorflow as tf
import numpy as np
import pdb
import cv2
import os
import glob
import net
from utils import dataset,create_dataset,file_processing,image_processing


def softmax(x,axis):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = axis, keepdims = True)
    s = x_exp / x_sum
    return s
def predict(models_path,image_dir,image_height, image_width,depth, char_set,captcha_size):
    '''

    :param models_path:
    :param image_dir:
    :param image_height:
    :param image_width:
    :param depth:
    :param char_set:
    :param captcha_size:
    :return:
    '''
    char_set_len = len(char_set)#字符种类个数
    X = tf.placeholder(tf.float32, [None, image_height, image_width,depth])
    keep_prob=1.0
    reg = None
    # output = net.simple_net(inputs=X, captcha_size=captcha_size, char_set_len=char_set_len,keep_prob=keep_prob)#learning_rate=0.001#学习率
    output, end_points= net.multilabel_nets(inputs=X, captcha_size=captcha_size, char_set_len=char_set_len,
                                            keep_prob=keep_prob, is_training=False, reg=reg)
    print("output.shape={}".format(output.get_shape()))


    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    for image_path in images_list:
        im = image_processing.read_image(image_path, image_height, image_width, normalization=True)
        im = im[np.newaxis, :]
        # pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_output = sess.run([output], feed_dict={X: im})

        predict = np.reshape(pre_output, newshape=[-1, captcha_size, char_set_len])
        predict = softmax(predict,axis=1)
        # max_score:[[0.999864  0.5114571 0.9802696 0.4393546]]
        max_idx_pre = np.argmax(predict, 2)
        max_idx_pre = np.squeeze(max_idx_pre)
        label_name=[]
        for index in max_idx_pre:
            label_name  += char_set[index]
        max_score = np.max(predict, axis=2)
        # print("{}:predict:{},max_idx_p:{},max_score:{},label_name:{}".format(image_path, predict,max_idx_pre,max_score,label_name))
        print("{},max_idx_p:{},max_score:{},label_name:{}".format(image_path,max_idx_pre,max_score,label_name))

    sess.close()



def predict2(models_path,image_dir,image_height, image_width,depth, captcha_size):
    '''

    :param models_path:
    :param image_dir:
    :param image_height:
    :param image_width:
    :param depth:
    :param captcha_size:
    :return:
    '''
    char_set_len = len(char_set)#字符种类个数
    X = tf.placeholder(tf.float32, [None, image_height, image_width,depth])
    keep_prob=1.0
    reg = None
    # output = net.simple_net(inputs=X, captcha_size=captcha_size, char_set_len=char_set_len,keep_prob=keep_prob)#learning_rate=0.001#学习率
    output, end_points= net.multilabel_nets(inputs=X, captcha_size=captcha_size, char_set_len=char_set_len,
                                            keep_prob=keep_prob, is_training=False, reg=reg)
    print("output.shape={}".format(output.get_shape()))

    predict = tf.reshape(output, [-1, captcha_size, char_set_len])
    predict = tf.nn.softmax(predict)
    #,max_score:[[0.999864  0.5114571 0.9802696 0.4393546]]
    max_idx_p = tf.argmax(predict, 2)
    max_score = tf.reduce_max(predict, reduction_indices=2)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    for image_path in images_list:
        im = image_processing.read_image(image_path, image_height, image_width, normalization=True)
        im = im[np.newaxis, :]
        # pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_id,predict_,max_score_ = sess.run([max_idx_p,predict,max_score], feed_dict={X: im})
        print("{}:pre_id:{},predict:{},max_score:{}".format(image_path, pre_id,predict_,max_score_))
    sess.close()

if __name__ == '__main__':
    # 载入字符集
    label_filename='./dataset/label_char_set.txt'
    char_set=file_processing.read_data(label_filename)
    # #
    batch_size=64
    image_height = 60
    image_width = 160
    depth=3
    captcha_size=4
    models_path='models/model-4500'
    image_dir='./dataset/test'

    predict(models_path,image_dir,image_height, image_width,depth, char_set,captcha_size)
