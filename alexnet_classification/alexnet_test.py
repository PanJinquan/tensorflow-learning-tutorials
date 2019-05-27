#coding=utf-8

import tensorflow as tf
import numpy as np
from datetime import datetime
from alexnet import *
import cv2
import os

def test(path):

    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32)
    output = alexnet(x, keep_prob)
    preds = tf.argmax(output, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './model/model.ckpt')
    for i in os.listdir(path):
        imgpath = os.path.join(path, i)
        im = cv2.imread(imgpath)
        im = cv2.resize(im, (224 , 224)) * (1. / 255)
        #cv2.imshow("img", im)
        im = np.expand_dims(im, axis=0)
        pred = sess.run(preds, feed_dict={x:im, keep_prob:1.0})
        #_cls = pred.argmax()
        #print pred
        print "{} image flowers class is: {}".format(i, pred)

        
        #cv2.waitKey()
        #cv2.destroyWindow('img')
    sess.close()


if __name__ == '__main__':
    path = './test'
    test(path)


    