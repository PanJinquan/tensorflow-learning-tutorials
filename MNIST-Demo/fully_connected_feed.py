import argparse
import os.path
import sys
import time
import os
from six.moves import xrange
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.examples.tutorials.mnist import mnist
import mnist
#过滤警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量，用来存放基本的模型(超)参数.
FLAGS = None

#产生 placeholder variables 来表达输入张量
def placeholder_inputs(batch_size):
  # 注意：placeholders的shape与整体数据集中图像集和标签集的shape是相互匹配的
  # 除了第一个维度现在是batch_size而不是训练集或测试集的全部大小
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

#填充 feed_dict 用于一个指定的训练阶段(given training step)
def fill_feed_dict(data_set, images_pl, labels_pl):
  #为参数指定的占位符创建feed_dict，用下一个批次的`batch size`个样本填充.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
  return feed_dict

#在给定的数据集上执行一次评估操作
def do_eval(sess, eval_correct,images_placeholder, labels_placeholder, data_set):
  # 运行一个回合(one epoch)的评估过程.
  true_count = 0  # 对预测正确的样本计数.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size #每个回合的执行步数
  num_examples = steps_per_epoch * FLAGS.batch_size  #样本总量
  #累加每个批次样本中预测正确的样本数量
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  #所有批次上的样本的精确度
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

#对MNIST网络训练指定的次数(一次训练称为一个training step)
def run_training():
    # 获取用于训练，验证和测试的图像数据以及类别标签集合
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    # 告诉TensorFlow模型将会被构建在默认的Graph上.
    with tf.Graph().as_default():
        # 为图像特征向量数据和类标签数据创建输入占位符
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # 从前向推断模型中构建用于预测的计算图
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # 为计算图添加计算损失的节点.
        loss = mnist.loss(logits, labels_placeholder)
        # 为计算图添加计算和应用梯度的训练节点
        train_op = mnist.training(loss, FLAGS.learning_rate)
        # 添加节点用于在评估过程中比较 logits 和 ground-truth labels .
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # 基于 TF collection of Summaries构建汇总张量.
        merged_summaries = tf.summary.merge_all()
        # 创建一个 saver 用于写入 训练过程中的模型的检查点文件(checkpoints).
        saver = tf.train.Saver()

        # 添加变量初始化节点(variable initializer Op).
        init = tf.global_variables_initializer()
        # 创建一个会话用来运行计算图中的节点
        sess = tf.Session()
        # 实例化一个 SummaryWriter 输出 summaries 和 Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        summary_writer.flush()

        # 运行初始化节点来初始化所有变量(Variables).
        sess.run(init)

        # 开启训练循环.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # 在此特定的训练步，使用真实的图像和类标签数据集填充 feed dictionary
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            # 在当前批次样本上把模型运行一步(run one step).
            # 返回值是从`train_op`和`loss`节点拿到的activations
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            # 计算当前批次的训练花费的时间
            duration = time.time() - start_time
            # 每隔100个批次就写入summaries并输出overview
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # 更新事件文件.还是调用sess.run()方法
                summary_str = sess.run(merged_summaries, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            # 周期性的保存一个检查点文件并评估当前模型的性能
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # 在所有训练集上评估模型.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                # 在验证集上评估模型.
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                # 在测试集上评估模型.
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

#创建日志文件夹,启动训练过程
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  #启动训练过程
  run_training()

#用ArgumentParser类把模型的(超)参数全部解析到全局变量FLAGS里面
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.1,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='MNIST_data/',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='logs/Fully_Connected_Feed',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  #把模型的(超)参数全部解析到全局变量FLAGS里面
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)