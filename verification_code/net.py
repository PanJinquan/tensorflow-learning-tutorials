# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : model.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-02 21:08:29
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 残差模块
def residual_block(net, reg, name):
    '''
    残差模块
    :param net:
    :return:
    '''
    input_nums = net.get_shape().as_list()[3]
    print("common_network inputs.shape:{}".format(net.get_shape()))
    with tf.variable_scope(name_or_scope=name):
        res = slim.conv2d(inputs=net,
                          num_outputs=128,
                          kernel_size=[1, 1],
                          padding="SAME",
                          scope="conv1",
                          activation_fn=tf.nn.relu,
                          weights_regularizer=reg)  # 1*1
        res = slim.conv2d(inputs=res,
                          num_outputs=64,
                          kernel_size=[3, 3],
                          padding="SAME",
                          scope="conv2",
                          activation_fn=tf.nn.relu,
                          weights_regularizer=reg)  # 3*3
        res = slim.conv2d(inputs=res,
                          num_outputs=input_nums,
                          kernel_size=[1, 1],
                          padding="SAME",
                          scope="conv3",
                          activation_fn=None,
                          weights_regularizer=reg)  # 1*1

        net = tf.nn.relu(tf.add(net, res))
    return net

batch_norm_params = {
      'decay': 0.96,
      'epsilon': 0.001,
      'scale': False,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
normalizer_fn = slim.batch_norm
normalizer_params = batch_norm_params
# normalizer_fn = None
# normalizer_params=None


def simple_net(inputs, captcha_size, char_set_len, keep_prob):
    '''

    :param inputs:
    :param captcha_size:
    :param char_set_len:
    :param keep_prob:
    :param is_training:
    :param reg:
    :return:
    '''
    print("inputs.shape:{}".format(inputs.get_shape()))
    net = slim.conv2d(inputs=inputs,
                      num_outputs=32,
                      kernel_size=[3, 3],
                      padding="SAME",
                      scope="conv1",
                      activation_fn=tf.nn.relu)
    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], stride=[2, 2], padding='SAME')
    net = slim.dropout(inputs=net, keep_prob=keep_prob)
    print("conv1.shape:{}".format(net.get_shape()))

    net = slim.conv2d(inputs=net,
                      num_outputs=64,
                      kernel_size=[3, 3],
                      padding="SAME",
                      scope="conv2",
                      activation_fn=tf.nn.relu)
    print("conv2.shape:{}".format(net.get_shape()))

    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], stride=[2, 2], padding='SAME')
    net = slim.dropout(inputs=net, keep_prob=keep_prob)

    net = slim.conv2d(inputs=net,
                      num_outputs=64,
                      kernel_size=[3, 3],
                      padding="SAME",
                      scope="conv3",
                      activation_fn=tf.nn.relu)
    print("conv3.shape:{}".format(net.get_shape()))

    net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], stride=[2, 2], padding='SAME')
    net = slim.dropout(inputs=net, keep_prob=keep_prob)

    net = slim.flatten(inputs=net)
    net = slim.fully_connected(inputs=net,
                               num_outputs=1024,
                               normalizer_fn=normalizer_fn,
                               normalizer_params=normalizer_params,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                               trainable=True,
                               activation_fn=tf.nn.relu, scope="fc_1")

    print("fc_1.shape:{}".format(net.get_shape()))

    out_w1 = tf.Variable(tf.truncated_normal(shape=[1024, captcha_size * char_set_len]))
    out_b1 = tf.Variable(tf.truncated_normal(shape=[captcha_size * char_set_len]))
    out = tf.matmul(net, out_w1) + out_b1
    print("out.shape:{}".format(out.get_shape()))
    # out = tf.nn.softmax(out)
    return out


def multilabel_nets(inputs, captcha_size, char_set_len, keep_prob, is_training, reg):
    '''

    :param inputs:
    :param captcha_size:
    :param char_set_len:
    :param keep_prob:
    :param is_training:
    :param reg:
    :return:
    '''
    with tf.variable_scope(name_or_scope='multilabel_nets', default_name='net', values=[inputs]) as sc:
        end_points_collection = sc.original_name_scope + 'end_points'#multilabel_nets/end_points
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],outputs_collections=end_points_collection):
            # slim.batch_norm, slim.dropout必须设置is_training参数，训练时True,测试时False
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                print("inputs.shape:{}".format(inputs.get_shape()))
                net = slim.conv2d(inputs=inputs,
                                  num_outputs=32,
                                  kernel_size=[3, 3],
                                  activation_fn=tf.nn.relu,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  weights_regularizer=reg,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  padding="SAME",
                                  scope="conv1")
                net=slim.max_pool2d(inputs=net,kernel_size=[2,2],stride=[2,2],padding='SAME')
                print("conv1.shape:{}".format(net.get_shape()))

                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[3, 3],
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  activation_fn=tf.nn.relu,
                                  weights_regularizer=reg,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  padding="SAME",
                                  scope="conv2")
                print("conv2.shape:{}".format(net.get_shape()))

                net=slim.max_pool2d(inputs=net,kernel_size=[2,2],stride=[2,2],padding='SAME')


                with tf.variable_scope('residual_block'):
                    net = residual_block(net, reg, name="res_1")
                    net = residual_block(net, reg, name="res_2")
                    net = residual_block(net, reg, name="res_3")
                    net = residual_block(net, reg, name="res_4")


                net = slim.conv2d(inputs=net,
                                  num_outputs=64,
                                  kernel_size=[3, 3],
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                  activation_fn=tf.nn.relu,
                                  weights_regularizer=reg,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  padding="SAME",
                                  scope="conv3")
                print("conv3.shape:{}".format(net.get_shape()))

                net=slim.max_pool2d(inputs=net,kernel_size=[2,2],stride=[2,2],padding='SAME')

                net = slim.flatten(inputs=net)
                net = slim.fully_connected(inputs=net,
                                           num_outputs=1024,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           weights_regularizer=reg,
                                           trainable=True,
                                           activation_fn=tf.nn.relu,
                                           normalizer_fn=normalizer_fn,
                                           normalizer_params=normalizer_params,
                                           scope="fc_1"
                                           )
                print("fc_1.shape:{}".format(net.get_shape()))
                net = slim.dropout(inputs=net,keep_prob=keep_prob)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                out_w1 = tf.Variable(tf.truncated_normal(shape=[1024, captcha_size * char_set_len]))
                out_b1 = tf.Variable(tf.truncated_normal(shape=[captcha_size * char_set_len]))
                out = tf.matmul(net, out_w1) + out_b1
                print("out.shape:{}".format(out.get_shape()))
                # out = tf.nn.softmax(out)
    return out,end_points