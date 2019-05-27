#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import os
from datetime import datetime
import slim.nets.vgg as vgg
from tf_read_images import *
import tensorflow.contrib.slim as slim


labels_nums = 2  # 类别个数
batch_size = 16  #
resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.float32, shape=[None, labels_nums], name='label')

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in xrange(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def train(train_filename,
          train_images_dir,
          train_log_step,
          train_param,
          val_filename,
          val_images_dir,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):
    '''
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step: 显示训练过程log信息间隔
    :param train_param: train参数
    :param val_record_file: 验证的tfrecord文件
    :param val_log_step: 显示验证过程log信息间隔
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_shape: 输入数据shape
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # # 从record中读取图片和labels数据
    tf_image, tf_labels = read_images(train_filename, train_images_dir, data_shape, shuffle=True,type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(tf_image, tf_labels, batch_size=batch_size, labels_nums=labels_nums,
                                                 one_hot=False, shuffle=True)

    # Define the model:
    with slim.arg_scope(vgg.vgg_arg_scope()):
        out, end_points = vgg.vgg_16(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training)

    loss=tf.reduce_sum(tf.squared_difference(x=out,y=input_labels))
    # loss1=tf.squared_difference(x=out,y=input_labels)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate=base_lr).minimize(loss)

    # tf.losses.add_loss(loss1)
    # # slim.losses.add_loss(my_loss)
    # loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
    # # Specify the optimization scheme:
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
    # # create_train_op that ensures that when we evaluate it to get the loss,
    # # the update_ops are done and the gradient updates are computed.
    # train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)
    

    saver = tf.train.Saver(max_to_keep=4)
    max_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps+1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images:batch_input_images,
                                                                      input_labels:batch_input_labels,
                                                                      keep_prob:0.5, is_training:True})
            if i % train_log_step == 0:
                print("%s: Step [%d]  train Loss : %f" % (datetime.now(), i, train_loss))
            # # train测试(这里仅测试训练集的一个batch)
            # if i%train_log_step == 0:
            #     train_acc = sess.run(accuracy, feed_dict={input_images:batch_input_images,
            #                                               input_labels: batch_input_labels,
            #                                               keep_prob:1.0, is_training: False})
            #     print "%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc)
            #
            # # val测试(测试全部val数据)
            # if i%val_log_step == 0:
            #     _, train_loss = sess.run([train_step, loss], feed_dict={input_images: batch_input_images,
            #                                                             input_labels: batch_input_labels,
            #                                                             keep_prob: 1.0, is_training: False})
            #     print "%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc)
            #
            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i %snapshot == 0 and i >0)or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix,i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # # 保存val准确率最高的模型
            # if mean_acc>max_acc and mean_acc>0.5:
            #     max_acc=mean_acc
            #     path = os.path.dirname(snapshot_prefix)
            #     best_models=os.path.join(path,'best_models_{}_{:.4f}.ckpt'.format(i,max_acc))
            #     print('------save:{}'.format(best_models))
            #     saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train_images_dir="../dataset/dataset_regression/images"
    train_filename ="../dataset/dataset_regression/train.txt"
    val_images_dir="../dataset/dataset_regression/images"
    val_filename ="../dataset/dataset_regression/val.txt"



    train_log_step=100
    base_lr = 0.001  # 学习率
    max_steps = 10000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=200
    snapshot=2000#保存文件间隔
    snapshot_prefix='models/model-regression.ckpt'

    train(train_filename=train_filename,
          train_images_dir=train_images_dir,
          train_log_step=train_log_step,
          train_param=train_param,
          val_filename=val_filename,
          val_images_dir=val_images_dir,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)