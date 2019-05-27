# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
import numpy as np

import tensorflow as tf

__all__ = [
  "train"
]

def train_no_bn(train_op,
                train_loss, train_acc,
                val_loss, val_acc,
                max_step,
                batch_size,
                train_examples,
                val_examples,
                train_log_step=100,
                val_log_step=500,
                snapshot=1000,
                snapshot_prefix = 'models/model.ckpt'):
  """训练函数
  Args:
    train_op: 训练`op`
    train_loss: 训练集计算误差的`op`
    train_acc: 训练集计算正确率的`op`
    val_loss: 验证集计算误差的`op`
    val_acc: 验证集计算正确率的`op`
    max_step: 最大迭代步长
    batch_sise: 一个批次中样本的个数
    train_log_step: 每隔多少步进行一次训练集信息输出
    val_log_step: 每隔多少步进行一次验证集信息输出
    
  Return:
    None
  """
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    try:
      _start = time.time()
      for step in xrange(max_step + 1):
        sess.run(train_op)
        if step % train_log_step == 0:
          _end = time.time()
          duration = _end - _start
          train_loss_, train_acc_ = sess.run([train_loss, train_acc])
          sec_per_batch = 1.0 * duration / train_log_step
          print('[train]: step %d loss = %.4f acc = %.4f (%.4f / batch)' % (step, train_loss_, train_acc_, sec_per_batch))
          _start = _end
          
        if step % val_log_step == 0:
          val_loss_, val_acc_ = sess.run([val_loss, val_acc])
          
          print('[val]: step %d loss = %.4f acc = %.4f' % (step, val_loss_, val_acc_))

        # 模型保存:每迭代snapshot次或者最后一次保存模型
        if (step % snapshot == 0 and step > 0) or step == max_step:
          print('-----save:{}-{}'.format(snapshot_prefix, step))
          saver.save(sess, snapshot_prefix, global_step=step)
      
      print('-------------------------Over all Result-------------------------')
      train_loss_, train_acc_ = _evaluation_no_bn(sess, train_loss, train_acc, train_examples, batch_size)
      print('[TRAIN]: loss = %.4f acc = %.4f' % (train_loss_, train_acc_))
      
      val_loss_, val_acc_ = _evaluation_no_bn(sess, val_loss, val_acc, val_examples, batch_size)
      print('[VAL]: loss = %.4f acc = %.4f' % (val_loss_, val_acc_))
          
    except tf.errors.OutOfRangeError:
      print('Epoch Limited. Done!')
    finally:
      coord.request_stop()
      
    coord.join(threads)
    
def _evaluation_no_bn(sess, loss_op, acc_op, num_examples, batch_size):
  '''
  :param sess:
  :param loss_op:
  :param acc_op:
  :param num_examples:
  :param batch_size:
  :return:
  '''
  max_steps = num_examples // batch_size
  losses = []
  accs = []
  for _ in xrange(max_steps):
    loss_value, acc_value = sess.run([loss_op, acc_op])
    losses.append(loss_value)
    accs.append(acc_value)
  
  mean_loss = np.array(losses, dtype=np.float32).mean()
  mean_acc = np.array(accs, dtype=np.float32).mean()
  
  return mean_loss, mean_acc


def train_with_bn(train_op, 
                  train_loss, train_acc,
                  val_loss, val_acc,
                  max_step, 
                  is_training,
                  batch_size,
                  train_examples,
                  val_examples,
                  train_log_step=100,
                  val_log_step=500,
                  snapshot=1000,
                  snapshot_prefix = 'models/model.ckpt'):
  """训练函数
  Args:
    train_op: 训练`op`
    train_loss: 训练集计算误差的`op`
    train_acc: 训练集计算正确率的`op`
    val_loss: 验证集计算误差的`op`
    val_acc: 验证集计算正确率的`op`
    max_step: 最大迭代步长
    is_training: bn层的参数
    train_examples: 训练样本个数
    val_examples: 验证样本个数
    train_batch: 训练集一个批次中样本的个数
    val_batch: 验证集一个批次中样本的个数
    train_log_step: 每隔多少步进行一次训练集信息输出
    val_log_step: 每隔多少步进行一次验证集信息输出
    
  Return:
    None
  """

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    try:
      _start = time.time()
      for step in xrange(max_step + 1):
        sess.run(train_op, feed_dict={is_training: True})
        if step % train_log_step == 0:
          _end = time.time()
          duration = _end - _start
          train_loss_, train_acc_ = sess.run([train_loss, train_acc], feed_dict={is_training: False})
          sec_per_batch = 1.0 * duration / train_log_step
          print('[train]: step %d loss = %.4f acc = %.4f (%.4f / batch)' % (step, train_loss_, train_acc_, sec_per_batch))
          train_losses.append(train_loss_)
          train_accs.append(train_acc_)
          _start = _end

        if step % val_log_step == 0:
          val_loss_, val_acc_ = sess.run([val_loss, val_acc], feed_dict={is_training: False})

          print('[val]: step %d loss = %.4f acc = %.4f' % (step, val_loss_, val_acc_))

          val_losses.append(val_loss_)
          val_accs.append(val_acc_)
        # 模型保存:每迭代snapshot次或者最后一次保存模型
        if (step % snapshot == 0 and step > 0) or step == max_step:
          print('-----save:{}-{}'.format(snapshot_prefix, step))
          saver.save(sess, snapshot_prefix, global_step=step)

      print('-------------------------Over all Result-------------------------')
      train_loss_, train_acc_ = _evaluation_with_bn(sess, train_loss, train_acc, is_training, train_examples, batch_size)
      print('[TRAIN]: loss = %.4f acc = %.4f' % (train_loss_, train_acc_))
      val_loss_, val_acc_ = _evaluation_with_bn(sess, val_loss, val_acc, is_training, val_examples, batch_size)
      print('[VAL]: loss = %.4f acc = %.4f' % (val_loss_, val_acc_))

    except tf.errors.OutOfRangeError:
      print('Epoch Limited. Done!')
    finally:
      coord.request_stop()

    coord.join(threads)
  
  return train_losses, train_accs, val_losses, val_accs
     
def _evaluation_with_bn(sess, loss_op, acc_op, is_training, num_examples, batch_size):
  '''

  :param sess:
  :param loss_op:
  :param acc_op:
  :param is_training:
  :param num_examples:
  :param batch_size:
  :return:
  '''
  max_steps = num_examples // batch_size
  losses = []
  accs = []
  for _ in xrange(max_steps):
    loss_value, acc_value = sess.run([loss_op, acc_op], feed_dict={is_training: False})
    losses.append(loss_value)
    accs.append(acc_value)
  
  mean_loss = np.array(losses, dtype=np.float32).mean()
  mean_acc = np.array(accs, dtype=np.float32).mean()
  return mean_loss, mean_acc
    
train = train_no_bn
