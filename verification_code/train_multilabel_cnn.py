# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : train.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-01 21:24:51
"""

import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import utils.image_processing as image_processing
import net
import tensorflow.contrib.slim as slim
from utils import dataset,create_dataset,file_processing

print("Tensorflow version:{}".format(tf.__version__))



def train(image_dir,data_filename,learning_rate,max_iter,step_log,val_log,snapshot,char_set,batch_size,image_height, image_width,depth, captcha_size):
    '''
    训练函数
    :param image_dir
    :param data_filename：#训练数据集的标注数据txt文本,eg:  1_4178.jpg 4 1 7 8
                                                        2_7915.jpg 7 9 1 5
                                                        3_2961.jpg 2 9 6 1
                                                        4_1765.jpg 1 7 6 5
    :param learning_rate：学习率
    :param max_iter：迭代次数
    :param step_log：训练log信息打印间隔
    :param val_log ：测试log信息打印间隔
    :param snapshot：模型保存间隔
    :param char_set ：验证码字符集
    :param batch_size
    :param image_height:
    :param image_width:
    :param depth:
    :param captcha_size: 验证码含有字符的个数
    :return:
    '''
    weight_decay=0.001 #正则化参数

    shape=[batch_size, image_height, image_width, depth]
    char_set_len = len(char_set)#字符种类个数

    # 载入训练数据，并分割成train和vall两部分
    images_list, labels_list=dataset.load_image_labels(data_filename)
    train_image_list, train_label_list, val_image_list, val_label_list=dataset.split_train_val_data(images_list, labels_list, factor=0.8)
    train_dataset_iterator=dataset.get_image_data(train_image_list, image_dir,train_label_list, batch_size,
                                         re_height=image_height, re_width=image_width, shuffle=False)
    val_dataset_iterator=dataset.get_image_data(val_image_list, image_dir,val_label_list, batch_size,
                                       re_height=image_height, re_width=image_width, shuffle=False)
    val_nums=len(val_image_list)


    X = tf.placeholder(tf.float32, [None, image_height, image_width,depth])
    Y = tf.placeholder(tf.float32, [None, captcha_size * char_set_len])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    is_training = tf.placeholder(tf.bool, name='is_training')

    reg = slim.l2_regularizer(scale=weight_decay)
    # output = net.simple_net(inputs=X, captcha_size=captcha_size, char_set_len=char_set_len,keep_prob=keep_prob)#learning_rate=0.001#学习率
    output, end_points= net.multilabel_nets(inputs=X, captcha_size=captcha_size, char_set_len=char_set_len,
                                            keep_prob=keep_prob, is_training=is_training, reg=reg)
    print("output.shape={}".format(output.get_shape()))


    # 定义loss函数
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))

    weights_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if weights_list and weight_decay is not None and weight_decay > 0:
        print("Regularization losses:")
        for i,rl in enumerate(weights_list):
            print("{}".format(rl.name))
        reg_loss=tf.reduce_sum(weights_list)#等价于sum(weights_list)
        total_loss = loss +tf.reduce_sum(reg_loss)
        #total_loss=reg_loss
    else:
      print("No regularization.")
      total_loss = loss


    # 优化器
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    # train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        print("update_ops:{}".format(update_ops))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    predict = tf.reshape(output, [-1, captcha_size, char_set_len])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, captcha_size, char_set_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        train_dataset = train_dataset_iterator.get_next()
        val_dataset = val_dataset_iterator.get_next()

        while True:
            # batch_x, batch_y = create_dataset.get_next_batch(char_set, shape=shape, captcha_size=captcha_size)
            batch_x, batch_y = sess.run(train_dataset)
            batch_y=dataset.multilabel2onehot(batch_y, captcha_size, char_set_len)

            _, total_loss_= sess.run([train_op, total_loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8,is_training:True})

            if step % step_log == 0:
                print("step:{},loss:{}".format(step, total_loss_))

            def net_evaluation(sess, accuracy, val_dataset, val_nums):
                '''
                val时，所有val数据都测试一遍
                :param sess:
                :param accuracy:
                :param val_dataset:
                :param val_nums:
                :return:
                '''
                val_max_steps = int(val_nums / batch_size)
                val_accs = []
                for _ in xrange(val_max_steps):
                    # batch_x_test, batch_y_test = create_dataset.get_next_batch(char_set, shape=shape, captcha_size=captcha_size)
                    batch_x_test, batch_y_test = sess.run(val_dataset)
                    batch_y_test = dataset.multilabel2onehot(batch_y_test, captcha_size, char_set_len)
                    val_acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0,is_training:False})
                    val_accs.append(val_acc)
                mean_acc = np.array(val_accs, dtype=np.float32).mean()
                return mean_acc


            # 每val_log计算一次准确率
            if step % val_log == 0:
                acc=net_evaluation(sess, accuracy, val_dataset, val_nums)
                print("------------------step:{},acc:{}".format(step, acc))

            if (step% snapshot == 0 and step > 0) or step== max_iter:
                snapshot_prefix = 'models/model'
                print('-----save:{}-{}'.format(snapshot_prefix, step))
                saver.save(sess, snapshot_prefix, global_step=step)

            if step==max_iter:
                break
            step += 1


if __name__ == '__main__':

    '''设置验证码的信息：
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    这里验证码字符集只考虑大写字母的情况,并且验证码大小captcha_size=4
    如果要预测大小写字母和数字，产生数据时，请加上number，alphabet和ALPHABET字符集
    '''
    # 载入字符集
    label_filename='./dataset/label_char_set.txt'
    char_set=file_processing.read_data(label_filename)

    # #
    batch_size=64
    image_height = 60
    image_width = 160
    depth=3
    captcha_size=4

    filename='./dataset/train.txt' #训练数据集的标注数据
    image_dir="./dataset/train"    #训练数据集图片的路径
    learning_rate=0.0001
    max_iter=100000
    step_log=10
    val_log=100
    snapshot=500
    train(image_dir,filename,learning_rate,max_iter,step_log,val_log,snapshot,char_set,batch_size,image_height, image_width,depth,captcha_size)
