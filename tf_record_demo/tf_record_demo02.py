# -*- coding: utf-8 -*-
# !/usr/bin/python3.5
# Author  : pan_jinquan
# Date    : 2018.6.29
# Function: image_convert to tfrecords
#############################################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


###############################################################################################
# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 生成实数型的属性
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# 显示图片
def show_image(image_name,image):
    # plt.figure("show_image")  # 图像窗口名称
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(image_name)  # 图像题目
    plt.show()

def load_labels_file(filename,labels_num=1):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签1，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        for line in f.readlines():
            im=line.split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(im[i+1]))
            images.append(im[0])
            labels.append(label)
    return images,labels

def read_image(filename, resize_height, resize_width):
    '''
    读取图片数据,
    '''
    # image = cv2.imread(filename)
    # image = cv2.resize(image, (resize_height, resize_width))
    # b, g, r = cv2.split(image)
    # rgb_image = cv2.merge([r, g, b])
    rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=rgb_image.resize((resize_width,resize_height))
    image=np.asanyarray(rgb_image)
    # show_image("src resize image",image)
    return image


def create_records(file, output_record_dir, resize_height, resize_width):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    :param file:输入保存图片信息的txt文件
    :param output_record_dir:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    '''
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_path, labels] in enumerate(zip(images_list, labels_list)):
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

def read_records(filename,resize_height, resize_width):
    '''
    解析record文件
    :param filename:
    :return:
    '''
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
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
    # tf_image=tf.reshape(tf_image, [-1]) # 转换为行向量
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # 设置图像的维度
    # tf_image = tf.cast(tf_image, tf.float32)
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5 # 归一化
    return tf_image, tf_height,tf_width,tf_depth,tf_label

def disp_records(record_file,resize_height, resize_width,show_nums=4):
    '''
    解析record文件，并显示show_nums张图片，主要用于验证生成record文件是否成功
    :param tfrecord_file: record文件路径
    :return:
    '''
    tf_image, tf_height, tf_width, tf_depth, tf_label = read_records(record_file,resize_height, resize_width)  # 读取函数
    # 显示前4个图片
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
        sess.close()


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
        # 关闭会话
        sess.close()

if __name__ == '__main__':
    # 参数设置
    train_file = 'train.txt'  # 图片路径
    output_record_dir = './tfrecords/my_record.tfrecords'
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    # 产生record文件
    create_records(train_file, output_record_dir, resize_height, resize_width)

    # 测试显示函数
    disp_records(output_record_dir,resize_height, resize_width)

    # batch_test(output_record_dir,resize_height, resize_width)


