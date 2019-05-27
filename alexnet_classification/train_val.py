#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import os
from datetime import datetime
from alexnet import *
from create_tf_record import *

labels_nums = 5  # 类别个数
batch_size = 32  #
resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
depths = 3
data_format = [batch_size, resize_height, resize_width, depths]

# 定义x为图片数据
x = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义y为labels数据
y = tf.placeholder(dtype=tf.float32, shape=[None, labels_nums], name='label')
keep_prob = tf.placeholder(tf.float32)

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in xrange(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy],feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

def train(train_record_file,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_format,
          snapshot,
          snapshot_prefix):
    '''
    :param train_record_file:
    :param train_log_step:
    :param train_param: train参数
    :param val_record_file:
    :param val_log_step:
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_format: 数据格式
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_format

    # 获得训练和测试的样本数
    train_nums=get_example_nums(train_record_file)
    val_nums=get_example_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # 从record中读取图片和labels数据
    # train数据
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width,type='normalization')  # 读取函数
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True,shuffle=True)
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width,type='normalization')  # 读取函数
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True,shuffle=False)


    # 建立网络，训练时keep_prob为0.5,测试时1.0
    output = alexnet(x, keep_prob, labels_nums)

    # 定义训练的loss函数。
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    #train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(loss)
    pred = tf.argmax(output, 1)#预测labels最大值下标
    truth = tf.argmax(y, 1)#实际labels最大值下标
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))

    saver = tf.train.Saver()
    max_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # 就是这一行
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps+1):
            batch_x, batch_y = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
            # train测试(这里仅测试训练集的一个batch)
            if i%train_log_step == 0:
                train_acc = sess.run(accuracy,feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
                print "%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc)

            # val测试(测试全部val数据)
            if i%val_log_step == 0:
                mean_loss, mean_acc=net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch,val_nums)
                print "%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc)

            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i %snapshot == 0 and i >0)or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix,i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            if mean_acc>=max_acc and mean_acc>0.5:
                max_acc=mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models=os.path.join(path,'best_models_{}_{:.4f}.ckpt'.format(i,max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    record_file='../dataset/dataset/record/train.tfrecords'
    train_log_step=100
    base_lr = 0.0001  # 学习率
    max_steps = 10000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=200
    snapshot=2000#保存文件间隔
    snapshot_prefix='models/model.ckpt'
    max_acc=0.998
    path=os.path.join('models', 'best_models_{}.ckpt'.format(max_acc))
    train(train_record_file=record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_format=data_format,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
