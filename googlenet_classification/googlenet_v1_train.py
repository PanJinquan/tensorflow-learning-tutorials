# -*-coding: utf-8 -*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : googlenet_v1_train.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-27 17:19:54
    @desc   : 训练文件
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from create_tf_record import *
import tensorflow.contrib.slim as slim
import tensorflow as tf
import googlenet_v1
import learning


labels_nums = 5  # 类别个数
batch_size = 32  #
resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
depths = 3
train_record_file = 'dataset/record/train.tfrecords'
val_record_file = 'dataset/record/train.tfrecords'


# 获得训练和测试的样本数
train_nums=get_example_nums(train_record_file)
val_nums=get_example_nums(val_record_file)
print('train nums:%d,val nums:%d'%(train_nums,val_nums))

# 从record中读取图片和labels数据
# train数据
train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=False, shuffle=True)
val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                      batch_size=batch_size, labels_nums=labels_nums,
                                                      one_hot=False, shuffle=False)

# 给卷积层设置默认的激活函数和`batch_norm`
with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm) as sc:
    conv_scope = sc
'''
    在训练的过程中, 所有`bn层`的`mean`和`variance`使用的是当前`batch`和之前`batch`的移动平均值
    而在预测的时候, 我们使用`bn层`本身的`mean`和`variance`.
    也就是说在训练和预测的时候, `bn层`是不同的
    所以训练集和验证集在`bn层`的参数不同, 用一个`placeholder`表示
'''

is_training = tf.placeholder(tf.bool, name='is_training')
with slim.arg_scope(conv_scope):
    train_out = googlenet_v1.googlenet(train_images_batch, labels_nums, is_training=is_training, verbose=True)
    val_out = googlenet_v1.googlenet(val_images_batch, labels_nums, is_training=is_training, reuse=True)

with tf.variable_scope('loss'):
    train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels_batch, logits=train_out, scope='train')
    val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels_batch, logits=val_out, scope='val')


with tf.name_scope('accuracy'):
    with tf.name_scope('train'):
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), train_labels_batch), tf.float32))
    with tf.name_scope('val'):
        val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_out, axis=-1, output_type=tf.int32), val_labels_batch), tf.float32))

# 学习率
lr = 0.01
# 优化方法
opt = tf.train.MomentumOptimizer(lr, momentum=0.9)

# 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
# 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
# 通过`tf.get_collection`获得所有需要更新的`op`
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
with tf.control_dependencies(update_ops):
    train_op = opt.minimize(train_loss)

learning.train_with_bn(train_op=train_op,
                  train_loss=train_loss,
                  train_acc=train_acc,
                  val_loss=val_loss,
                  val_acc=val_acc,
                  max_step=20000,
                  is_training=is_training,
                  batch_size=batch_size,
                  train_examples=train_nums,
                  val_examples=val_nums,
                  train_log_step=100,
                  val_log_step=500,
                  snapshot=1000,
                  snapshot_prefix = 'models/model.ckpt')
