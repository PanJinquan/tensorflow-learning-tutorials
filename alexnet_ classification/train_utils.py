#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
from datetime import datetime
from alexnet import *



def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32)# * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    #print 'images的样子是：', img
    #print 'label的样子是：', label
    #pdb.set_trace()
    return img, label

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
    # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # 设置图像的维度
    # 存储的图像类型为uint8,这里需要将类型转为tf.float32
    tf_image = tf.cast(tf_image, tf.float32)
    # [1]若需要归一化请使用:
    # tf_image = tf.cast(tf_image, tf.float32) * (1. / 255)       # 归一化
    # [2]若需要归一化,且中心化,假设均值为0.5,请使用:
    # tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 # 中心化
    # return tf_image, tf_height,tf_width,tf_depth,tf_label
    return tf_image,tf_label


def train(record_file,train_param,class_nums,data_format):
    [base_lr,max_steps,snapshot,snapshot_prefix,display]=train_param
    [batch_size,resize_height,resize_width,channels]=data_format

    # 训练数据
    train_images,train_labels = read_records(record_file,resize_height, resize_width) # 读取函数

    train_images_batch, train_labels_batch = tf.train.shuffle_batch([train_images, train_labels],
                                                    batch_size=batch_size,
                                                    capacity=392,
                                                    min_after_dequeue=200)

    # 测试数据
    val_images,val_labels = read_records(record_file,resize_height, resize_width) # 读取函数

    val_images_batch, val_labels_batch = tf.train.shuffle_batch([val_images, val_labels],
                                                    batch_size=batch_size,
                                                    capacity=392,
                                                    min_after_dequeue=200)

    # 通过 alexnet 构建训练和测试的输出


    train_out = alexnet(train_images_batch,0.5,class_nums)

    # 注意当再次调用 alexnet 函数时, 如果要使用之前调用时产生的变量值, **必须**要重用变量域
    # val_out = alexnet(val_images_batch,1.0,class_nums,reuse=True)

    # #### 定义损失函数
    # 这里真实的 labels 不是一个 one_hot 型的向量, 而是一个数值, 因此我们使用
    with tf.variable_scope('loss'):
        train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels_batch, logits=train_out, scope='train')
        val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels_batch, logits=val_out, scope='val')

    # #### 定义正确率`op`
    with tf.name_scope('accuracy'):
        with tf.name_scope('train'):
            train_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), train_labels_batch), tf.float32))
        with tf.name_scope('train'):
            val_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(val_out, axis=-1, output_type=tf.int32), val_labels_batch), tf.float32))

    # #### 构造训练`op`
    base_lr = 0.0001

    # opt = tf.train.MomentumOptimizer(base_lr, momentum=0.9)
    # train_op = opt.minimize(train_loss)

    #train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(train_loss)

    # 在这里我们已经将训练过程封装好了, 感兴趣的同学可以进入`train.py`查看
    from utils.learning import train

    # train(train_op, train_loss, train_acc, val_loss, val_acc, max_step=10000, batch_size=batch_size)
    snapshot = 1000
    snapshot_prefix = 'models/model.ckpt'
    max_step = 10000
    train(train_op, train_loss, train_acc, val_loss, val_acc, snapshot, snapshot_prefix, max_step, batch_size=batch_size,
          train_log_step=50, val_log_step=200)

    # 可以看到, 20000步的训练后, `AlexNet`在训练集和测试集分别达到了`0.97`和`0.72`的准确率


if __name__ == '__main__':
    record_file='dataset/record/record.tfrecords'

    base_lr = 0.0001  # 学习率
    max_steps = 1000  # 迭代次数
    display=50  #每迭代50次,输出loss和准确率
    snapshot=500#保存文件间隔
    snapshot_prefix='./models/model.ckpt'
    train_param=[base_lr,max_steps,snapshot,snapshot_prefix,display]

    class_nums = 5  # 类别个数
    batch_size = 32  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    channels=3
    data_format=[batch_size,resize_height,resize_width,channels]
    train(record_file,train_param,class_nums,data_format)
