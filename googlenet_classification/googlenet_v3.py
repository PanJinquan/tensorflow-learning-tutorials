# coding: UTF-8
import tensorflow as tf
import time
import math
from datetime import datetime

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# Inception V3 model has 42 layers
# part one
def inception_v3_arg_scope(weight_decay=0.00004,  # l2 decacy_rate
						   stddev=0.1,
						   batch_norm_var_collection='moving_vars'):
    # 定义batch normalization的参数字典
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }

    # slim.arg_scope给函数的参数自动赋值
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=trunc_normal(stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as sc:
            return sc


# part two # inception_v3_base生成网络的卷积部分
def inception_v3_base(inputs, scope=None):
    end_points = {}
    # inputs:输入图片数据的tensor, scope为包含了函数默认参数的环境 end_points保存某些关键节点

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        ''' 5 conv2d +  3 max_pool2d '''
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # 非Inception Module的卷积层
            # slim.conv2d(tensor, 通道, kernal size, stride, padding)
            # inputs:299x299x3
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            # 149x149x32
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            # 147x147x32
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            # 147x147x64

            net = slim.max_pool2d(net, [3, 3], stride=2, scope='Maxpool_3a_3x3')
            # 73x73x64

            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            # 73x73x80
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            # 71x71x192

            net = slim.max_pool2d(net, [3, 3], stride=2, scope='Maxpool_5a_3x3')
        # 35x35x192

        # 三个连续的Inception模块组
        #########################''' 3 inception blocks '''########################
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            '''*---the first block:5b~5d---*'''
            '''mixed1:35x35x256'''
            # 第一个模块组的第一个Inception Module
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 64+64+96+32 = 256

            # mixed2: 35x35x288
            # 第一个模块组的第二个Inception Module
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 64+64+96+64 = 288

            # mixed3:35x35x288
            # 第一个模块组的第三个Inception Module
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 64+64+96+64 = 288

            # 第二个Inception模块组包括5个Inception Module
            # 第一个Inception Module
            '''*---the second block:6a~6e---*'''
            '''mixed4:17x17x768'''
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    # 35x35x96
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                # 17x17x96
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='Maxpool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], axis=3)
            # channel = 384+96+288 = 768

            # 第二个Inception Module
            '''mixed5:17x17x768'''
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 192*4 = 768

            # 第三个Inception Module
            '''mixed6:17x17x768'''
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 192*4 = 768

            # 第四个Inception Module 与第三个完全一致
            # 通过Inception module的结构增加卷积和非线性 提炼特征
            '''mixed7:17x17x768'''
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 192*4 = 768

            # 第五个Inception Module
            '''mixed8:17x17x768'''
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 192*4 = 768
            end_points['Mixed_6e'] = net

            # 第三个Inception模块组包括3个Inception Module
            # 第一个Inception Module
            '''*---the third block:7a~7c---*'''
            '''mixed9:8x8x1280'''
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 8x8x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='Maxpool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], axis=3)
            # channel = 320+192+768 = 1280

            # 第二个Inception Module
            '''mixed10:8x8x2048'''
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), \
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], axis=3)
                # 8x8x768
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), \
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0c_3x1')], axis=3)
                # 8x8x768
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 320+768+768 + 192 = 2048

            # 第三个Inception Module
            '''mixed11:8x8x2048'''
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), \
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], axis=3)
                # 8x8x768
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), \
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], axis=3)
                # 8x8x768
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
            # channel = 320+768+768 + 192 = 2048

            return net, end_points


# part three #全局平均池化、SoftMax、 Auxiliary logits
def inception_v3(inputs,
				 num_classes=1000,
				 is_training=True,
				 dropout_keep_prob=0.7,
				 prediction_fn=slim.softmax,  # 分类的函数
				 spatial_squeeze=True,  # 对输出进行squeeze操作
				 reuse=None,
				 scope='Inceptionv3'):
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

            net, end_points = inception_v3_base(inputs, scope=scope)

            # Auxiliary Logits:辅助分类的节点
            # aux:processing
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5],
                                             weights_initializer=trunc_normal(0.01),
                                             padding='VALID', scope='Conv2d_2a_5x5')

                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                                             scope='Conv2d_2b_1x1')
                    if spatial_squeeze:  # tf.squeeze消除tensor中前两个为1的维度
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

            # net:processing
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net

                feats_64 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_dr_1x1')
                end_points['feats_64'] = feats_64
                logits = slim.conv2d(feats_64, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                     scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


# def cross_entropy(logits,label_true,num_classes):
# 	y = tf.one_hot(tf.cast(label_true,tf.int32), num_classes, on_value = 1, off_value = 0)
# 	loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.cast(y,tf.float32), logits = logits)
# 	loss = tf.reduce_mean(loss)
# 	# loss = -tf.reduce_sum(y*tf.log(logits))
# 	return loss

# # pretrained_network = '/home/liesmars/Softwares/tf-rcnn/lib/model/inception_v3.ckpt'
# # pretrained_vars = pretrained_network.all_params
# def optimizer(loss, lr, current_epoch):
# 	# train_opt = tf.train.GradientDescentOptimizer(lr_rate).minimize(loss,global_step = current_epoch)
# 	train_opt =  tf.train.RMSPropOptimizer(lr).minimize(loss,global_step = current_epoch)
# 	return train_opt


# def cal_accuracy(predictions, label, num_classes):
#     y_ = np.argmax(predictions,axis = 1)[0]
#     y_ = tf.one_hot(tf.cast(y_,tf.int32),num_classes, on_value=1, off_value=0)
#     y = tf.one_hot(tf.cast(label,tf.int32),num_classes, on_value=1, off_value=0)
#     acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_,tf.int32),tf.cast(y,tf.int32)),tf.float32))
#     return acc

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration= %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    # 计算每轮迭代的平均耗时mn和标准差sd
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


if __name__ == '__main__':
    batch_size = 32
    height, width = 299, 299
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(inception_v3_arg_scope()):
        logits, end_points = inception_v3(inputs, is_training=False)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num_batches = 100
    time_tensorflow_run(sess, logits, 'Forward')

