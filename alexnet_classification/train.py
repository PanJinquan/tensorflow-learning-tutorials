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

    # 定义x为图片数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, resize_height,resize_width,channels], name='input')
    # 定义y为labels数据
    y = tf.placeholder(dtype=tf.float32, shape=[None, class_nums], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = alexnet(x, keep_prob, class_nums)
    #probs = tf.nn.softmax(output)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    #train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(loss)

    pred = tf.argmax(output, 1)
    truth = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))

    # 从record中读取图片和labels数据
    # images, labels = read_and_decode('./train.tfrecords')
    train_images,train_labels = read_records(record_file,resize_height, resize_width) # 读取函数
    train_images_batch, train_labels_batch = tf.train.shuffle_batch([train_images, train_labels],
                                                    batch_size=batch_size,
                                                    capacity=392,
                                                    min_after_dequeue=200)
    train_labels_batch = tf.one_hot(train_labels_batch, class_nums, 1, 0)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps+1):
            batch_x, batch_y = sess.run([train_images_batch, train_labels_batch])
#            print batch_x, batch_x.shape
#            print batch_y
#            pdb.set_trace()
            _, train_loss = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
            # train测试
            if i%display == 0:
                train_acc = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
                print "%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc)


            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i %snapshot == 0 and i >0)or i == max_steps:
                print('save:{}-{}'.format(snapshot_prefix,i))
                #checkpoint_path = os.path.join(FLAGS.train_dir, './model/model.ckpt')
                saver.save(sess, snapshot_prefix, global_step=i)

        coord.request_stop()
        coord.join(threads)


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
