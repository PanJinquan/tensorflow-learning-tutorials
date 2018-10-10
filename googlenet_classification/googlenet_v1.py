#-*-coding:utf-8-*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : googlenet_v1.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-10 10:24:11
"""

# 我们用`tf-slim`构建网络模型
import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
一个 inception 模块的四个并行线路如下：
1.一个 1 x 1 的卷积，一个小的感受野进行卷积提取特征
2.一个 1 x 1 的卷积加上一个 3 x 3 的卷积，1 x 1 的卷积降低输入的特征通道，减少参数计算量，然后接一个 3 x 3 的卷积做一个较大感受野的卷积
3.一个 1 x 1 的卷积加上一个 5 x 5 的卷积，作用和第二个一样
4.一个 3 x 3 的最大池化加上 1 x 1 的卷积，最大池化改变输入的特征排列，1 x 1 的卷积进行特征提取
'''
def inception(x, d0_1, d1_1, d1_3, d2_1, d2_5, d3_1, scope='inception', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # 我们把`slim.conv2d`,`slim.max_pool2d`的默认参数放在`slim`的参数域里
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
            # 第一个分支
            with tf.variable_scope('branch0'):
                branch_0 = slim.conv2d(x, d0_1, [1, 1], scope='conv_1x1')

            # 第二个分支
            with tf.variable_scope('branch1'):
                branch_1 = slim.conv2d(x, d1_1, [1, 1], scope='conv_1x1')
                branch_1 = slim.conv2d(branch_1, d1_3, [3, 3], scope='conv_3x3')

            # 第三个分支
            with tf.variable_scope('branch2'):
                branch_2 = slim.conv2d(x, d2_1, [1, 1], scope='conv_1x1')
                branch_2 = slim.conv2d(branch_2, d2_1, [5, 5], scope='conv_5x5')

            # 第四个分支
            with tf.variable_scope('branch3'):
                branch_3 = slim.max_pool2d(x, [3, 3], scope='max_pool')
                branch_3 = slim.conv2d(branch_3, d3_1, [1, 1], scope='conv_1x1')

            # 连接
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)

            return net


# 使用`inception`模块去构建整个`googlenet`
def googlenet(inputs, num_classes, reuse=None, is_training=None, verbose=False):
    with tf.variable_scope('googlenet', reuse=reuse):
        # 给`batch_norm`的`is_training`参数设定默认值.
        # `batch_norm`和`is_training`密切相关, 当`is_trainig=True`时,
        # 它使用的是一个`batch`数据的移动平均,方差值
        # 当`is_training=Fales`时, 它使用的是固定值
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding='SAME', stride=1):
                net = inputs
                with tf.variable_scope('block1'):
                    net = slim.conv2d(net, 64, [5, 5], stride=2, scope='conv_5x5')

                    if verbose:
                        print('block1 output: {}'.format(net.shape))

                with tf.variable_scope('block2'):
                    net = slim.conv2d(net, 64, [1, 1], scope='conv_1x1')
                    net = slim.conv2d(net, 192, [3, 3], scope='conv_3x3')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='max_pool')

                    if verbose:
                        print('block2 output: {}'.format(net.shape))

                with tf.variable_scope('block3'):
                    net = inception(net, 64, 96, 128, 16, 32, 32, scope='inception_1')
                    net = inception(net, 128, 128, 192, 32, 96, 64, scope='inception_2')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='max_pool')

                    if verbose:
                        print('block3 output: {}'.format(net.shape))

                with tf.variable_scope('block4'):
                    net = inception(net, 192, 96, 208, 16, 48, 64, scope='inception_1')
                    net = inception(net, 160, 112, 224, 24, 64, 64, scope='inception_2')
                    net = inception(net, 128, 128, 256, 24, 64, 64, scope='inception_3')
                    net = inception(net, 112, 144, 288, 24, 64, 64, scope='inception_4')
                    net = inception(net, 256, 160, 320, 32, 128, 128, scope='inception_5')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='max_pool')

                    if verbose:
                        print('block4 output: {}'.format(net.shape))

                with tf.variable_scope('block5'):
                    net = inception(net, 256, 160, 320, 32, 128, 128, scope='inception1')
                    net = inception(net, 384, 182, 384, 48, 128, 128, scope='inception2')
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope='avg_pool')

                    if verbose:
                        print('block5 output: {}'.format(net.shape))

                with tf.variable_scope('classification'):
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='logit')

                    if verbose:
                        print('classification output: {}'.format(net.shape))

                return net
