############################################################################################
# !/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Author  : zhaoqinghui
# Date    : 2016.5.10
# Function: image convert to tfrecords
#############################################################################################

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# 参数设置
###############################################################################################
train_file = 'train.txt'  # 训练图片
# name = 'train'  # 生成train.tfrecords
# output_directory = './tfrecords'
output_record_dir= './tfrecords/my_record.tfrecords'
resize_height = 100  # 存储图片高度
resize_width = 100  # 存储图片宽度


###############################################################################################
#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#生成实数型的属性
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#显示图片
def show_image(image_name,image):
    # plt.figure("show_image")  # 图像窗口名称
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(image_name)  # 图像题目
    plt.show()

#载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签，如：test_image/1.jpg 0
def load_txt_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, delimiter=" ", dtype=[('col1', 'S120'), ('col2', 'i8')])
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example.decode('ascii'))
        labels.append(label)
    return np.asarray(examples), np.asarray(labels), len(lines)

#读取原始图像数据
def read_image(filename, resize_height, resize_width):
    # image = cv2.imread(filename)
    # image = cv2.resize(image, (resize_height, resize_width))
    # b, g, r = cv2.split(image)
    # rgb_image = cv2.merge([r, g, b])
    rgb_image=Image.open(filename)
    rgb_image=rgb_image.resize((resize_width,resize_height))
    image=np.asanyarray(rgb_image)
    show_image("src resize image",image)
    return image


def transform2tfrecord(train_file, output_record_dir, resize_height, resize_width):
    _examples, _labels, examples_num = load_txt_file(train_file)
    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        print('No.%d' % (i))
        image = read_image(example, resize_height, resize_width)
        print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def disp_tfrecords(tfrecord_list_file):
    filename_queue = tf.train.string_input_producer([tfrecord_list_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
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
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # print(repr(image))
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)
    init_op = tf.initialize_all_variables()
    resultImg = []
    resultLabel = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(4):
            image_eval = image.eval()
            resultLabel.append(label.eval())
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            resultImg.append(image_eval_reshape)
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
            show_image("decode_from_tfrecords",image_eval_reshape)
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel


def read_tfrecord(filename_queuetemp):
    filename_queue = tf.train.string_input_producer([filename_queuetemp])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image
    tf.reshape(image, [256, 256, 3])
    # normalize
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # label
    label = tf.cast(features['label'], tf.int32)
    return image, label


def test():
    transform2tfrecord(train_file, output_record_dir, resize_height, resize_width)  # 转化函数
    # img, label = disp_tfrecords(output_record_dir)  # 显示函数
    img, label = read_tfrecord(output_record_dir)  # 读取函数
    print(img,label)
    # show_image("image",img)


if __name__ == '__main__':
    test()
