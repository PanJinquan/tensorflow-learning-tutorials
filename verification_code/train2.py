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

from utils import create_dataset

def train(char_set,batch_size,image_height, image_width,depth, captcha_size):
    '''
    训练函数
    :param char_set
    :param batch_size
    :param image_height:
    :param image_width:
    :param depth
    :param captcha_size: 验证码含有字符的个数
    :return:
    '''
    shape=[batch_size, image_height, image_width, depth]
    char_set_len = len(char_set)#字符种类个数

    X = tf.placeholder(tf.float32, [None, image_height, image_width,depth])
    Y = tf.placeholder(tf.float32, [None, captcha_size * char_set_len])
    keep_prob = tf.placeholder(tf.float32)  # dropout

    # inputs = tf.reshape(X, shape=[-1, image_height, image_width, 1])
    inputs=X
    output = net.simple_net(inputs, captcha_size, char_set_len, keep_prob)
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, captcha_size, char_set_len])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, captcha_size, char_set_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = create_dataset.get_next_batch(char_set, shape=shape, captcha_size=captcha_size)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("step:{},loss:{}".format(step, loss_))

            # 每100 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = create_dataset.get_next_batch(char_set, shape=shape, captcha_size=captcha_size)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("------------------step:{},acc:{}".format(step, acc))
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
            if step==50000:
                break
            step += 1


if __name__ == '__main__':
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    '''设置验证码的信息：
    这里验证码字符集只考虑数字的情况,并且验证码大小captcha_size=4
    '''
    char_set=number
    # char_set = number + alphabet + ALPHABET # 如果要预测26个大小写字母，则加上alphabet和ALPHABET字符集
    batch_size=32
    image_height = 60
    image_width = 160
    depth=3
    captcha_size=4
    # 产生一个验证码样本并显示
    text, image = create_dataset.gen_captcha_text_and_image(char_set=char_set,
                                                     captcha_height=image_height,
                                                     captcha_width=image_width,
                                                     captcha_size=captcha_size)
    image_processing.cv_show_image(text, image)#显示验证码
    print("验证码图像shape:{}".format(image.shape))  # (60, 160, 3)
    print("验证码字符个数:{}".format(captcha_size))

    # 训练
    train(char_set,batch_size,image_height, image_width,depth,captcha_size)
