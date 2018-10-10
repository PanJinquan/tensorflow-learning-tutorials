
# coding: utf-8

# # InceptionNet
# 前面我们讲的 VGG 是 2014 年 ImageNet 比赛的亚军，那么冠军是谁呢？就是我们马上要讲的 InceptionNet，
# 这是 Google 的研究人员提出的网络结构(所以也叫做`GoogLeNet`)，在当时取得了非常大的影响，
# 因为网络的结构变得前所未有，它颠覆了大家对卷积网络的串联的印象和固定做法，
# 采用了一种非常有效的 inception 模块，得到了比 VGG 更深的网络结构，
# 但是却比 VGG 的参数更少，因为其去掉了后面的全连接层，所以参数大大减少，同时有了很高的计算效率。
# 
# ![](https://ws2.sinaimg.cn/large/006tNc79ly1fmprhdocouj30qb08vac3.jpg)
# 
# 这是 googlenet 的网络示意图，下面我们介绍一下其作为创新的 inception 模块。

# ## Inception 模块
# 在上面的网络中，我们看到了多个四个并行卷积的层，这些四个卷积并行的层就是 inception 模块，可视化如下
# 
# ![](https://ws4.sinaimg.cn/large/006tNc79gy1fmprivb2hxj30dn09dwef.jpg)

# 一个 inception 模块的四个并行线路如下：
# 1.一个 1 x 1 的卷积，一个小的感受野进行卷积提取特征
# 2.一个 1 x 1 的卷积加上一个 3 x 3 的卷积，1 x 1 的卷积降低输入的特征通道，减少参数计算量，然后接一个 3 x 3 的卷积做一个较大感受野的卷积
# 3.一个 1 x 1 的卷积加上一个 5 x 5 的卷积，作用和第二个一样
# 4.一个 3 x 3 的最大池化加上 1 x 1 的卷积，最大池化改变输入的特征排列，1 x 1 的卷积进行特征提取
# 
# 最后将四个并行线路得到的特征在通道这个维度上拼接在一起，下面我们可以实现一下

# In[ ]:

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from utils import cifar10_input

batch_size = 64
train_imgs, train_labels, val_imgs, val_labels = cifar10_input.load_data(data_dir='cifar10_data/')

# 我们用`tf-slim`构建网络模型
import tensorflow.contrib.slim as slim


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
        # 当`is_training=True`时, 它使用的是固定值
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


# 给卷积层设置默认的激活函数和`batch_norm`


with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm) as sc:
    conv_scope = sc


'''
    在训练的过程中, 所有`bn层`的`mean`和`variance`使用的是当前`batch`和之前`batch`的移动平均值
    而在预测的时候, 我们使用`bn层`本身的`mean`和`variance`.
    也就是说在训练和预测的时候, `bn层`是不同的
    所以训练集和验证集在`bn层`的参数不同, 用一个`placeholder`表示
'''


is_training = tf.placeholder(tf.bool, name='is_training')

with slim.arg_scope(conv_scope):
    train_out = googlenet(train_imgs, 10, is_training=is_training, verbose=True)
    val_out = googlenet(val_imgs, 10, is_training=is_training, reuse=True)



with tf.variable_scope('loss'):
    train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_out, scope='train')
    val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels, logits=val_out, scope='val')



with tf.name_scope('accuracy'):
    with tf.name_scope('train'):
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), train_labels), tf.float32))
    with tf.name_scope('val'):
        val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_out, axis=-1, output_type=tf.int32), val_labels), tf.float32))



lr = 0.01

opt = tf.train.MomentumOptimizer(lr, momentum=0.9)


# 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数, 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新


# 通过`tf.get_collection`获得所有需要更新的`op`
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
with tf.control_dependencies(update_ops):
    train_op = opt.minimize(train_loss)


# 在这里, 我们把带`bn层`的训练过程封装在`utils.learning.train_with_bn`中, 感兴趣的同学可以看看


from utils.learning import train_with_bn


train_with_bn(train_op, train_loss, train_acc, val_loss, val_acc, 20000, is_training)


# `InceptionNet`有很多的变体, 比如`InceptionV1`,`V2`, `V3`, `V4`版本, 尝试查看论文, 自己动手实现一下并比较他们的不同
