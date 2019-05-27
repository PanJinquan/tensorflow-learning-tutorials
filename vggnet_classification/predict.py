#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
from datetime import datetime
import alexnet
import cv2
import os
import glob

def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image


def  predict(models_path,image_dir,labels_filename,class_nums, data_format):
    [batch_size, resize_height, resize_width, channels] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32)
    output = alexnet.alexnet(x, keep_prob, class_nums)
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
        pre_score,pre_label = sess.run([score,class_id], feed_dict={x:im, keep_prob:1.0})
        max_score=pre_score[0,pre_label]
        print "{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score)
    sess.close()


if __name__ == '__main__':

    class_nums=5
    image_dir='test_image'
    labels_filename='dataset/label.txt'
    models_path='models/best_models_10000_0.9992.ckpt'

    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    channels=3
    data_format=[batch_size,resize_height,resize_width,channels]
    predict(models_path,image_dir, labels_filename, class_nums, data_format)
