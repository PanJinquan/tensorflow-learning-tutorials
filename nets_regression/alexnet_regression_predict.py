#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import nets.alexnet as alexnet
from create_tf_record_multi_label import *
import tensorflow.contrib.slim as slim
import utils.statistic as statistic


def  predict(models_path,image_dir,labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    # labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    # Define the model:
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        out, end_points = alexnet.alexnet_v2(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0,is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别


    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score = sess.run([out], feed_dict={input_images:im})
        mean_std=statistic.get_score(pre_score,type="mean_std")
        # print("image_path:{},pre_score:{},mean_std:{}".format(image_path,pre_score,mean_std))

        print("image_path:{},mean_std:{}".format(image_path,mean_std))

    sess.close()


if __name__ == '__main__':

    class_nums=10
    image_dir='test_image/image3'
    models_path='models/model-regression.ckpt-306608'

    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    predict(models_path,image_dir, class_nums, data_format)
