import tensorflow as tf

# 声明两个变量
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
init_op = tf.global_variables_initializer() # 初始化全部变量
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1) # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    sess.run(init_op)
    print("v1:", sess.run(v1)) # 打印v1、v2的值一会读取之后对比
    print("v2:", sess.run(v2))
    saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
    print("Model saved in file:", saver_path)