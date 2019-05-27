#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
from datetime import datetime
import cv2
import os
import glob
import googlenet_v1
from create_tf_record import *
import tensorflow.contrib.slim as slim




def  predict(models_path,image_dir,labels_filename,class_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    x = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
    # 给卷积层设置默认的激活函数和`batch_norm`
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm) as sc:
        conv_scope = sc
    with slim.arg_scope(conv_scope):
        # train_out = googlenet(train_images_batch, labels_nums, is_training=is_training, verbose=True)
        # val_out = googlenet(val_images_batch, labels_nums, is_training=is_training, reuse=True)
        output = googlenet_v1.googlenet(x, class_nums, is_training=False,reuse=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(output)
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        # im = cv2.imread(imgpath)
        # im = cv2.resize(im, (224 , 224))# * (1. / 255)
        # im = np.expand_dims(im, axis=0)
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={x:im})
        max_score=pre_score[0,pre_label]
        print "{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score)
    sess.close()


if __name__ == '__main__':

    class_nums=5
    image_dir='test_image'
    labels_filename='../dataset/dataset/label.txt'
    models_path='models/model.ckpt-3000'

    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    predict(models_path,image_dir, labels_filename, class_nums, data_format)
