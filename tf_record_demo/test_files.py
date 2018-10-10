#-*-coding:utf-8-*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : test_files.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-06 15:17:08
"""

import  tensorflow as tf

filenames= "dataset/record/record0.*"
files = tf.train.match_filenames_once(filenames)
init = tf.local_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(files ))
# filename_queue=tf.train.string_input_producer(files,shuffle=False)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(files))