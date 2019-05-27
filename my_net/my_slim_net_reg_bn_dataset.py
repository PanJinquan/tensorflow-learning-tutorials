#coding=utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.create_tf_record import *
from datetime import datetime

labels_nums = 5  # 类别个数
batch_size = 16  #
resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def im_pre(image):
    image=tf.image.random_brightness(image,max_delta=63)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_contrast(image,lower=0.2,upper=1.8)
    image=tf.image.per_image_standardization(image)
    return  image


# 残差模块
def residual_block(net,num_outputs):
    tmp=slim.conv2d(inputs=net,
                    num_outputs=64,
                    kernel_size=[1,1],
                    padding="SAME",
                    scope="conv1")# 1*1
    tmp=slim.conv2d(inputs=tmp,
                    num_outputs=64,
                    kernel_size=[3,3],
                    padding="SAME",
                    scope="conv2")# 3*3
    tmp=slim.conv2d(inputs=tmp,
                    num_outputs=num_outputs,
                    kernel_size=[1,1],
                    padding="SAME",
                    scope="conv3")# 1*1
    # net=tf.add(net,tmp)
    net=net+tmp
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
# normalizer_params = None
# define nets
def nets(inputs, num_classes, dropout_keep_prob, is_training,reg):
    with tf.variable_scope(name_or_scope="my_net", default_name='net', values=[inputs]) as sc:
        # slim.batch_norm, slim.dropout必须设置is_training参数，训练时True,测试时False
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            end_points_collection = "my_net"
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                # slim.convolution2d=slim.conv2d
                net = slim.conv2d(
                    inputs=inputs,
                    num_outputs=32,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                    weights_regularizer=reg,
                    kernel_size=(5, 5),
                    activation_fn=tf.nn.relu,
                    stride=(1, 1),
                    padding="SAME",
                    trainable=True,
                    scope="conv_1")
                net = slim.max_pool2d(net,kernel_size=[3, 3],stride=[2, 2],padding="SAME")
                net = slim.conv2d(
                    inputs=net,
                    num_outputs=32,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=reg,
                    kernel_size=(5, 5),
                    activation_fn=tf.nn.relu,
                    stride=(1, 1),
                    padding="SAME",
                    trainable=True,
                    scope="conv_2")
                net = slim.max_pool2d(net,kernel_size=[3, 3],stride=[2, 2],padding="SAME")
                with tf.variable_scope('residual_block'):
                    net=residual_block(net,num_outputs=32)

                net = slim.conv2d(
                    inputs=net,
                    num_outputs=64,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=reg,
                    kernel_size=(5, 5),
                    activation_fn=tf.nn.relu,
                    stride=(1, 1),
                    padding="SAME",
                    trainable=True,
                    scope="conv_3")
                net = slim.max_pool2d(net,kernel_size=[3, 3],stride=[2, 2], padding="SAME")
                # pool_3_flat = tf.reshape(pool_3, [-1, 4 * 4 * 64])
                net = slim.flatten(inputs=net)
                net = slim.fully_connected(inputs=net,
                                             num_outputs=1024,
                                             normalizer_fn=normalizer_fn,
                                             normalizer_params=normalizer_params,
                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                             weights_regularizer=reg,
                                             trainable=True,
                                             activation_fn=tf.nn.relu, scope="fc_1")
                net = slim.dropout(net, keep_prob=dropout_keep_prob)

                net = slim.fully_connected(inputs=net,
                                             num_outputs=128,
                                             normalizer_fn=normalizer_fn,
                                             normalizer_params=normalizer_params,
                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                             weights_regularizer=reg,
                                             trainable=True,
                                           activation_fn=tf.nn.relu, scope="fc_2")
                net = slim.dropout(net, keep_prob=dropout_keep_prob)

                # 将collection转化为python的dict,这样net节点名称都保存在end_points中
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                out_w1 = tf.Variable(tf.truncated_normal(shape=[128, num_classes]))
                out_b1 = tf.Variable(tf.truncated_normal(shape=[num_classes]))
                net = tf.matmul(net, out_w1) + out_b1
    return net


def train(train_record_file,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):
    '''
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step: 显示训练过程log信息间隔
    :param train_param: train参数
    :param val_record_file: 验证的tfrecord文件
    :param val_log_step: 显示验证过程log信息间隔
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_shape: 输入数据shape
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # 获得训练和测试的样本数
    train_nums=get_example_nums(train_record_file)
    val_nums=get_example_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # 从record中读取图片和labels数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=False)
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)

    reg = slim.l2_regularizer(scale=0.1)
    out = nets(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training, reg=reg)
    print("combine.shape={}".format(out.get_shape()))
    # tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)#添加交叉熵损失loss=1.6

    # pred = tf.cast(tf.argmax(tf.nn.softmax(out), 1), tf.int32)
    weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    reg_ws = slim.apply_regularization(reg, weights_list=weight)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    loss1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=out))
    loss = loss1 + tf.reduce_sum(reg_ws)# 不加正则项loss<100,加上正则项loss>10000
    tf.summary.scalar("loss",loss)

    # train_op = tf.train.AdamOptimizer(base_lr).minimize(loss)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(base_lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, input_labels), tf.float32))
    tf.summary.scalar("accuracy",accuracy)
    merged=tf.summary.merge_all()

    train_writer=tf.summary.FileWriter('./log',tf.get_default_graph())

    saver = tf.train.Saver()
    max_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps+1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images:batch_input_images,
                                                                      input_labels:batch_input_labels,
                                                                      keep_prob:0.8, is_training:True})
            # train测试(这里仅测试训练集的一个batch)
            if i%train_log_step == 0:
                train_acc,train_summary = sess.run([accuracy,merged], feed_dict={input_images:batch_input_images,
                                                          input_labels: batch_input_labels,
                                                          keep_prob:1.0, is_training: False})
                train_writer.add_summary(train_summary,i)
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc))

            # val测试(测试全部val数据)
            if i%val_log_step == 0:
                mean_loss, mean_acc=net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch,val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i %snapshot == 0 and i >0)or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix,i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            # if mean_acc>max_acc and mean_acc>0.5:
            #     max_acc=mean_acc
            #     path = os.path.dirname(snapshot_prefix)
            #     best_models=os.path.join(path,'best_models_{}_{:.4f}.ckpt'.format(i,max_acc))
            #     print('------save:{}'.format(best_models))
            #     saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)
    train_writer.close()


if __name__ == '__main__':
    train_record_file='../dataset/dataset/record/train.tfrecords'
    val_record_file='../dataset/dataset/record/val.tfrecords'

    train_log_step=100
    base_lr = 0.0001  # 学习率
    max_steps = 10000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=200
    snapshot=2000#保存文件间隔
    snapshot_prefix='models/model.ckpt'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)