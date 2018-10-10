#coding=utf-8

from datetime import datetime
import tensorflow as tf
import math
import time
import numpy as np

def print_activations(t):
    print t.op.name, ' ', t.get_shape().as_list()

def alexnet(images,_dropout,labels_nums,reuse=None):
    with tf.variable_scope('AlexNet', reuse=reuse):
        parameters = []
        "Conv1 layer"
        with tf.name_scope('conv1') as scope:
            weight = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                dtype=tf.float32, stddev=0.1), name='weights')
            conv = tf.nn.conv2d(images, weight, [1, 4, 4, 1], padding='SAME')
            bias = tf.Variable(tf.constant(0., shape=[64], dtype=tf.float32),
                                trainable=True, name='bias')
            conv1 = tf.nn.relu(conv + bias, name=scope)
            print_activations(conv1)
            parameters += [weight, bias]

        "LRN layer + Maxpool layer"
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, [1, 3, 3, 1], [1, 2, 2, 1],padding='VALID', name='pool1')
        print_activations(pool1)

        "Conv2 layer"
        with tf.name_scope('conv2') as scope:
            weight = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                dtype=tf.float32, stddev=0.1), name='weights')
            conv = tf.nn.conv2d(pool1, weight, [1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(tf.constant(0., shape=[192], dtype=tf.float32),
                                trainable=True, name='bias')
            conv2 = tf.nn.relu(conv + bias, name=scope)
            print_activations(conv2)
            parameters += [weight, bias]

        "LRN layer + Maxpool layer"
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool2')
        print_activations(pool2)

        "Conv3 layer"
        with tf.name_scope('conv3') as scope:
            weight = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                dtype=tf.float32, stddev=0.1), name='weights')
            conv = tf.nn.conv2d(pool2, weight, [1, 1, 1,1], padding='SAME')
            bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[384]),
                                trainable=True, name='bias')
            conv3 = tf.nn.relu(conv + bias, name=scope)
            print_activations(conv3)
            parameters += [weight, bias]

        "Conv4 layer"
        with tf.name_scope('conv4') as scope:
            weight = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                dtype=tf.float32, stddev=0.1), name='weights')
            conv = tf.nn.conv2d(conv3, weight, [1, 1, 1,1], padding='SAME')
            bias = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32),
                                trainable=True, name='bias')
            conv4 = tf.nn.relu(conv + bias, name=scope)
            print_activations(conv4)
            parameters +=[weight, bias]

        "Conv5 layer"
        with tf.name_scope('conv5') as scope:
            weight = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                dtype=tf.float32, stddev=0.1), name='weights')
            conv = tf.nn.conv2d(conv4, weight, [1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(tf.constant(0., shape=[256], dtype=tf.float32),
                                trainable=True, name='bias')
            conv5 = tf.nn.relu(conv + bias, name=scope)
            print_activations(conv5)
            parameters += [weight, bias]

        "Maxpool layer"
        pool5 = tf.nn.max_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool5')
        print_activations(pool5)

        "FC1 layer"
        flatten = tf.reshape(pool5, [-1, 6*6*256])
        with tf.name_scope('fc1') as scope:
            weight = tf.Variable(tf.truncated_normal([6*6*256, 4096], mean=0, stddev=0.01))
            bias   = tf.Variable(tf.constant(0., shape=[4096], dtype=tf.float32),
                                trainable=True, name='bias')
            fc1 = tf.nn.relu(tf.matmul(flatten, weight) + bias, name=scope)
            print_activations(fc1)
            parameters += [weight, bias]

        "dropout layer"
        dropout1 = tf.nn.dropout(fc1, _dropout)

        "FC2 layer"
        with tf.name_scope('fc2') as scope:
            weight = tf.Variable(tf.truncated_normal([4096,4096], mean=0, stddev=0.01))
            bias   = tf.Variable(tf.constant(0., shape=[4096], dtype=tf.float32),
                                trainable=True, name='bias')
            fc2 = tf.nn.relu(tf.matmul(dropout1, weight) + bias, name=scope)
            print_activations(fc2)
            parameters += [weight, bias]

        "dropout layer"
        dropout2 = tf.nn.dropout(fc2, _dropout)

        "FC3 layer"
        with tf.name_scope('fc3') as scope:
            weight = tf.Variable(tf.truncated_normal([4096, labels_nums], mean=0, stddev=0.01))
            bias   = tf.Variable(tf.constant(0., shape=[labels_nums], dtype=tf.float32),
                                trainable=True, name='bias')
            output = tf.nn.bias_add(tf.matmul(dropout2, weight), bias)
            print_activations(output)

        return output



