# -*-coding: utf-8 -*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : check_ava_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-15 10:47:30
"""

import numpy as np
import os
import glob

import tensorflow as tf

'''
Checks all images from the AVA dataset if they have corrupted jpegs, and lists them for removal.

Removal must be done manually !
'''
def parse_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def check_image(images_list,labels_list,isRemove=True):
    '''
    检测图像是否损坏
    :param images_list:
    :param labels_list:
    :param isRemove:
    :return:
    '''
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        fn = tf.placeholder(dtype=tf.string)
        img = parse_data(fn)
        i = 0
        while i < len(images_list):
            path = images_list[i]
            try:
                sess.run(img, feed_dict={fn: path})
            except Exception as e:
                print(" Bad image:{}".format(path))
                if isRemove:
                    os.remove(path)
                    print(" info:----------------remove image:{}".format(path))
                images_list.pop(i)
                labels_list.pop(i)
                continue
            i += 1
            if i%1000==0:
                print(" processing:step={},current path={}".format(i,path))
    # while i<len(images_list):#判断图像文件是否损坏(比较耗时)
    #     try:
    #         Image.open(path).verify()
    #     except :
    #         print(" Bad image:{}".format(path))
    #         if isRemove:
    #             os.remove(path)
    #             print(" info:----------------remove image:{}".format(path))
    #         images_list.pop(i)
    #         labels_list.pop(i)
    #         continue
    #     i += 1
    return images_list, labels_list

def isValidImage(images_list,labels_list,sizeTh=1000,isRemove=False):
    ''' 去除不存的文件和文件过小的文件列表
    :param images_list:
    :param labels_list:
    :param sizeTh: 文件大小阈值,单位：字节B，默认1000B
    :param isRemove: 是否在硬盘上删除被损坏的原文件
    :return:
    '''
    i=0
    while i<len(images_list):
        path=images_list[i]
        # 判断文件是否存在
        if not (os.path.exists(path)):
            print(" non-existent file:{}".format(path))
            images_list.pop(i)
            labels_list.pop(i)
            continue
        # 判断文件是否为空
        if os.path.getsize(path)<sizeTh:
            print(" empty file:{}".format(path))
            if isRemove:
                os.remove(path)
                print(" info:----------------remove image:{}".format(path))
            images_list.pop(i)
            labels_list.pop(i)
            continue
        i += 1
    # 检测图像是否损坏(比较耗时，进行一次检测并删除后，这一步就可以不执行了)
    images_list, labels_list=check_image(images_list,labels_list,True)
    return images_list, labels_list

###############################
def get_ava_data_label(txt_filename):
    images_list=[]
    labels_list=[]
    with open(txt_filename) as f:
        lines = f.readlines()
        for line in lines:
            #rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content=line.rstrip().split(' ')
            name=content[1]+'.jpg'
            labels=[]
            for value in content[2:12]:
                labels.append(float(value))
            images_list.append(name)
            labels_list.append(labels)
    return images_list,labels_list


def load_image_labels(test_files):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签1，如：test_image/1.jpg 0 2
    :param test_files:
    :return:
    '''
    images_list=[]
    labels_list=[]
    with open(test_files) as f:
        lines = f.readlines()
        for line in lines:
            #rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content=line.rstrip().split(' ')
            name=content[0]
            labels=[]
            for value in content[1:]:
                labels.append(float(value))
            images_list.append(name)
            labels_list.append(labels)
    return images_list,labels_list

if __name__=="__main__":
    images_dir = '/home/ubuntu/project/SmartAlbum/AVA_dataset/dataset/images'
    filename = '/home/ubuntu/project/SmartAlbum/aesthetic_dataset/dataset_mean_std/AVA.txt'
    images_list, labels_list = get_ava_data_label(filename)
    images_list = list(map(lambda x: os.path.join(images_dir, x), images_list))
    images_list, labels_list=isValidImage(images_list, labels_list,1000,True)
    print('files nums:%d'%(len(images_list)))
