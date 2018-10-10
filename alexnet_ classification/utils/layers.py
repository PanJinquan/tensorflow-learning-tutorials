# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

__all__ = [
  "conv", 
  "max_pool", 
  "fc"]

def variable_weight(shape, stddev=5e-2):
    init = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(shape=shape, initializer=init, name='weight')

def variable_bias(shape):
    init = tf.constant_initializer(0.1)
    return tf.get_variable(shape=shape, initializer=init, name='bias')

def conv(x, ksize, out_depth, strides, padding='SAME', act=tf.nn.relu, scope='conv_layer', reuse=None):
    """构造一个卷积层
    Args:
        x: 输入
        ksize: 卷积核的大小, 一个长度为2的`list`, 例如[3, 3]
        output_depth: 卷积核的个数
        strides: 卷积核移动的步长, 一个长度为2的`list`, 例如[2, 2]
        padding: 卷积核的补0策略
        act: 完成卷积后的激活函数, 默认是`tf.nn.relu`
        scope: 这一层的名称(可选)
        reuse: 是否复用
    
    Return:
        out: 卷积层的结果
    """
    # 这里默认数据是NHWC输入的
    in_depth = x.get_shape().as_list()[-1]
    
    with tf.variable_scope(scope, reuse=reuse):
        # 先构造卷积核
        shape = ksize + [in_depth, out_depth]
        with tf.variable_scope('kernel'):
            kernel = variable_weight(shape)
            
        strides = [1, strides[0], strides[1], 1]
        # 生成卷积
        conv = tf.nn.conv2d(x, kernel, strides, padding, name='conv')
        
        # 构造偏置
        with tf.variable_scope('bias'):
            bias = variable_bias([out_depth])
            
        # 和偏置相加
        preact = tf.nn.bias_add(conv, bias)
        
        # 添加激活层
        out = act(preact)
        
        return out
        
def max_pool(x, ksize, strides, padding='SAME', name='pool_layer'):
    """构造一个最大值池化层
    Args:
        x: 输入
        ksize: pooling核的大小, 一个长度为2的`list`, 例如[3, 3]
        strides: pooling核移动的步长, 一个长度为2的`list`, 例如[2, 2]
        padding: pooling的补0策略
        name: 这一层的名称(可选)
    
    Return:
        pooling层的结果
    """
    out = tf.nn.max_pool(x, [1, ksize[0], ksize[1], 1], [1, strides[0], strides[1], 1], padding, name=name)
    
    return out
    
def fc(x, out_depth, act=tf.nn.relu, scope='fully_connect', reuse=None):
    """构造一个全连接层
    Args:
        x: 输入
        out_depth: 输出向量的维数
        act: 激活函数, 默认是`tf.nn.relu`
        scope: 名称域, 默认是`fully_connect`
        reuse: 是否需要重用
    """
    in_depth = x.get_shape().as_list()[-1]
    
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('weight'):
            weight = variable_weight([in_depth, out_depth])
            
        with tf.variable_scope('bias'):
            bias = variable_bias([out_depth])
            
        fc = tf.nn.bias_add(tf.matmul(x, weight), bias, name='fc')
        out = act(fc)
        
        return out
        
def batch_norm_1d(x, is_training, decay=0.1, scope='bn'):
    eps = 1e-5
    
    with tf.variable_scope(scope):
        moving_mean = tf.get_variable('moving_mean', initializer=tf.zeros_initializer(), shape=x.get_shape()[-1:], dtype=tf.float32, trainable=False)
        moving_var = tf.get_variable('moving_var', initializer=tf.zeros_initializer(), shape=x.get_shape()[-1:], dtype=tf.float32, trainable=False)
        
        tf.add_to_collection('moving_mean', moving_mean)
        tf.add_to_collection('moving_var', moving_var)

        gamma = tf.get_variable('gamma', initializer=tf.random_normal_initializer(), shape=x.get_shape()[-1:])
        beta = tf.get_variable('beta', initializer=tf.random_normal_initializer(), shape=x.get_shape()[-1:])
        
        def batch_norm_train():
            x_mean, x_var = tf.nn.moments(x, axes=[0])
            x_hat = (x - x_mean) / tf.sqrt(x_var + eps)
            update_moving_mean = moving_mean.assign(decay * moving_mean + (1 - decay) * x_mean)
            update_moving_var = moving_var.assign(decay * moving_var + (1 - decay) * x_var)
            update_moving_op = tf.group(update_moving_mean, update_moving_var)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_op)
            
            return x_hat
        
        def batch_norm_test():
            x_hat = (x - moving_mean) / tf.sqrt(moving_var + eps)

            return x_hat
        
        x_hat = tf.cond(is_training, batch_norm_train, batch_norm_test)
        
        return gamma * x_hat + beta
