#coding=utf-8
# 单隐层SoftMax Regression分类器：训练和保存模型模块
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import graph_util

print('tensortflow:{0}'.format(tf.__version__))

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

# create model
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 输入节点名：x_input
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')
with tf.name_scope('layer'):
    with tf.name_scope('W'):
        # tf.zeros([3, 4], tf.int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        W = tf.Variable(tf.zeros([784, 10]), name='Weights')
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([10]), name='biases')
    with tf.name_scope('W_p_b'):
        Wx_plus_b = tf.add(tf.matmul(x, W), b, name='Wx_plus_b')

    y = tf.nn.softmax(Wx_plus_b, name='final_result')

# 定义损失函数和优化方法
with tf.name_scope('loss'):
    loss = -tf.reduce_sum(y_ * tf.log(y))
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    print(train_step)
# 初始化
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
# 训练
for step in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
    # variables = tf.all_variables()
    # print(len(variables))
    # print(sess.run(b))

# 测试模型准确率
pre_num = tf.argmax(y, 1, output_type='int32', name="output")  # 输出节点名：output
correct_prediction = tf.equal(pre_num, tf.argmax(y_, 1, output_type='int32'))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
a = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
print('测试正确率：{0}'.format(a))

# 保存训练好的模型
# 形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
with tf.gfile.FastGFile('model/mnist.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())
sess.close()
# 注：
