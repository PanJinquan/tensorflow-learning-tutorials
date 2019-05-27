# -*-coding: utf-8 -*-
"""
    @Project: tf_record_demo
    @File   : tf_record_batchSize.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-27 17:19:54
    @desc   : 将图片数据保存为多个record文件
"""

##########################################################################

import tensorflow as tf
import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt
import random
from PIL import Image


##########################################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 生成实数型的属性
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

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

def load_labels_file(filename,labels_num=1):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        for lines in f.readlines():
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels

def read_image(filename, resize_height, resize_width):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :return: 返回的图片数据是uint8,[0,255]
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
    # show_image("src resize image",image)

    return rgb_image


def create_records(image_dir,file, record_txt_path, batchSize,resize_height, resize_width):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param output_record_txt_dir:保存record文件的路径
    :param batchSize: 每batchSize个图片保存一个*.tfrecords,避免单个文件过大
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    '''
    if os.path.exists(record_txt_path):
        os.remove(record_txt_path)

    setname, ext = record_txt_path.split('.')

    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1)
    sample_num = len(images_list)
    # 打乱样本的数据
    # random.shuffle(labels_list)
    batchNum = int(math.ceil(1.0 * sample_num / batchSize))

    for i in range(batchNum):
        start = i * batchSize
        end = min((i + 1) * batchSize, sample_num)
        batch_images = images_list[start:end]
        batch_labels = labels_list[start:end]
        # 逐个保存*.tfrecords文件
        filename = setname + '{0}.tfrecords'.format(i)
        print('save:%s' % (filename))

        writer = tf.python_io.TFRecordWriter(filename)
        for i, [image_name, labels] in enumerate(zip(batch_images, batch_labels)):
            image_path=os.path.join(image_dir,batch_images[i])
            if not os.path.exists(image_path):
                print('Err:no image',image_path)
                continue
            image = read_image(image_path, resize_height, resize_width)
            image_raw = image.tostring()
            print('image_path=%s,shape:( %d, %d, %d)' % (image_path,image.shape[0], image.shape[1], image.shape[2]),'labels:',labels)
            # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
            label=labels[0]
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'height': _int64_feature(image.shape[0]),
                'width': _int64_feature(image.shape[1]),
                'depth': _int64_feature(image.shape[2]),
                'label': _int64_feature(label)
            }))
            writer.write(example.SerializeToString())
        writer.close()

        # 用txt保存*.tfrecords文件列表
        # record_list='{}.txt'.format(setname)
        with open(record_txt_path, 'a') as f:
            f.write(filename + '\n')

def read_records(filename,resize_height, resize_width):
    '''
    解析record文件
    :param filename:保存*.tfrecords文件的txt文件路径
    :return:
    '''
    # 读取txt中所有*.tfrecords文件
    with open(filename, 'r') as f:
        lines = f.readlines()
        files_list=[]
        for line in lines:
            files_list.append(line.rstrip())

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer(files_list,shuffle=False)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据

    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # 设置图像的维度
    # 存储的图像类型为uint8,这里需要将类型转为tf.float32
    # tf_image = tf.cast(tf_image, tf.float32)
    # [1]若需要归一化请使用:
    tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)# 归一化
    # tf_image = tf.cast(tf_image, tf.float32) * (1. / 255)  # 归一化
    # [2]若需要归一化,且中心化,假设均值为0.5,请使用:
    # tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 #中心化
    return tf_image, tf_height,tf_width,tf_depth,tf_label

def disp_records(record_file,resize_height, resize_width,show_nums=4):
    '''
    解析record文件，并显示show_nums张图片，主要用于验证生成record文件是否成功
    :param tfrecord_file: record文件路径
    :param resize_height:
    :param resize_width:
    :param show_nums: 默认显示前四张照片

    :return:
    '''
    tf_image, tf_height, tf_width, tf_depth, tf_label = read_records(record_file,resize_height, resize_width)  # 读取函数
    # 显示前show_nums个图片
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image,height,width,depth,label = sess.run([tf_image,tf_height,tf_width,tf_depth,tf_label])  # 在会话中取出image和label
            # image = tf_image.eval()
            # 直接从record解析的image是一个向量,需要reshape显示
            # image = image.reshape([height,width,depth])
            print('shape:',image.shape,'label:',label)
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
            show_image("image:%d"%(label),image)
        coord.request_stop()
        coord.join(threads)


def batch_test(record_file,resize_height, resize_width):
    '''
    :param record_file: record文件路径
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batch一般作为网络的输入
    '''

    tf_image,tf_height,tf_width,tf_depth,tf_label = read_records(record_file,resize_height, resize_width) # 读取函数

    # 使用shuffle_batch可以随机打乱输入:
    # shuffle_batch用法:https://blog.csdn.net/u013555719/article/details/77679964
    min_after_dequeue = 100#该值越大,数据越乱,必须小于capacity
    batch_size = 4
    # capacity = (min_after_dequeue + (num_threads + a small safety margin∗batchsize)
    capacity = min_after_dequeue + 3 * batch_size#容量:一个整数,队列中的最大的元素数

    image_batch, label_batch = tf.train.shuffle_batch([tf_image, tf_label],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, label_batch])
            # 这里仅显示每个batch里第一张图片
            show_image("image", images[0, :, :, :])
            print(images.shape, labels)
        # 停止所有线程
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # 参数设置
    image_dir='dataset'
    train_file = 'dataset/train.txt'  # 图片路径
    output_record_txt = 'dataset/record/record.txt'#指定保存record的文件列表
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    batchSize=8000     #batchSize一般设置为8000,即每batchSize张照片保存为一个record文件
    # 产生record文件
    create_records(image_dir=image_dir,
                   file=train_file,
                   record_txt_path=output_record_txt,
                   batchSize=batchSize,
                   resize_height=resize_height,
                   resize_width=resize_width)

    # 测试显示函数
    disp_records(output_record_txt,resize_height, resize_width)

    # batch_test(output_record_txt,resize_height, resize_width)
