# -*-coding: utf-8 -*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : tf_read_images.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-15 10:47:30
"""
import tensorflow as tf
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def show_image(title,image):
    '''
    显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')    # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()

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
            if i % 100 == 0:
                print(" processing:step={},current path={}".format(i, path))

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

def isValidImage(images_list,labels_list,sizeTh=1000,isRemove=False,detectImage=False):
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
    if detectImage:
        images_list, labels_list=check_image(images_list,labels_list)
    return images_list, labels_list


def read_images(filename,images_dir,reshape,shuffle=False,type=None):
    '''
    读取图像
    :param filename:TXT文件路径
    :param images_dir:图片文件的路径
    :param reshape=[batch_size,h,w,depth]:
    :param shuffle
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         standardization:标准化float32-[0,1],再减均值中心化
    :return:
    '''
    batch_size,height,width,channels=reshape
    images_list, labels_list = load_image_labels(filename)
    images_list = list(map(lambda x: os.path.join(images_dir, x), images_list))
    images_list, labels_list=isValidImage(images_list, labels_list,1000,True,False)
    # print(images_list, labels_list)
    # 获得训练和测试的样本数
    print('files nums:%d'%(len(images_list)))

    # [1]生成队列
    image_que, labels_que = tf.train.slice_input_producer([images_list, labels_list], shuffle=shuffle)
    tf_image = tf.image.decode_jpeg(tf.read_file(image_que), channels=channels)

    # [2]图像处理
    # image=tf.image.rgb_to_grayscale(image)
    tf_image = tf.image.resize_images(tf_image, (height, width))#会将uint8转为float32

    # [3]数据类型处理
    # 存储的图像类型为uint8,tensorflow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type == 'normalization':  # [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        # tf_image = tf.cast(tf_image, dtype=tf.uint8)
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type == 'standardization': # 标准化
        # tf_image = tf.cast(tf_image, dtype=tf.uint8)
        # tf_image = tf.image.per_image_standardization(tf_image)  # 标准化（减均值除方差）
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5  # 中心化
    return  tf_image,labels_que


def get_batch_images(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False):
    '''
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                        batch_size=batch_size,
                                                        capacity=capacity)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch,labels_batch

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

def batch_test(filename,images_dir):
    shuffle = False
    labels_nums = 2
    batch_size = 5
    height = 224
    width = 224
    channels = 3
    reshape = [batch_size, height, width, channels]
    tf_image, tf_labels = read_images(filename, images_dir, reshape, shuffle=shuffle,type='normalization')
    image_batch, labels_batch = get_batch_images(tf_image, tf_labels, batch_size=batch_size, labels_nums=labels_nums,
                                                 one_hot=False, shuffle=shuffle)
    with tf.Session() as sess:  # 开始一个会话
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, labels_batch])
            # 这里仅显示每个batch里第一张图片
            show_image("image", images[2, :, :, :])
            print('shape:{},tpye:{},labels:{}'.format(images.shape, images.dtype, labels))

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

if __name__=="__main__":
    # images_dir="/home/ubuntu/project/tfTest/tensorflow-learning-tutorials/dataset/dataset_regression/test_image"
    # filename="/home/ubuntu/project/tfTest/tensorflow-learning-tutorials/dataset/dataset_regression/test_image.txt"

    images_dir = '/home/ubuntu/project/tfTest/tensorflow-learning-tutorials/dataset/dataset_regression/test_image'
    filename = '/home/ubuntu/project/tfTest/tensorflow-learning-tutorials/dataset/dataset_regression/test_image.txt'

    # ava数据集测试
    # images_dir="/home/ubuntu/project/SmartAlbum/AVA_dataset/dataset/images"
    # filename ="../dataset/ava_dataset/val.txt"
    batch_test(filename,images_dir)


