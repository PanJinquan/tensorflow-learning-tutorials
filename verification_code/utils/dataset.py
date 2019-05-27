# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-03 18:45:13
"""
import tensorflow as tf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from utils import file_processing,image_processing
from sklearn import preprocessing

print("TF Version:{}".format(tf.__version__))

resize_height = 0  # 指定存储图片高度
resize_width = 0  # 指定存储图片宽度

def load_image_labels(filename):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签1，如：test_image/1.jpg 0 2
    :param filename:
    :return:
    '''
    images_list=[]
    labels_list=[]
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            #rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content=line.rstrip().split(' ')
            name=content[0]
            labels=[]
            for value in content[1:]:
                labels.append(int(value))
            images_list.append(name)
            labels_list.append(labels)
    return images_list,labels_list

def split_train_val_data(image_list,label_list,factor=0.8):
    trian_num=int(len(image_list)*factor)
    train_image_list = image_list[:trian_num]
    train_label_list = label_list[:trian_num]

    val_image_list = image_list[trian_num:]
    val_label_list = label_list[trian_num:]
    print("data info***************************")
    print("--train nums:{}".format(len(train_image_list)))
    print("--val   nums:{}".format(len(val_image_list)))
    print("************************************")
    return train_image_list,train_label_list,val_image_list,val_label_list

def show_image(title, image):
    '''
    显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()

def tf_resize_image(image, width=0, height=0):
    if (width is None) or (height is None):  # 错误写法：resize_height and resize_width is None
        return image
    image = tf.image.resize_images(image, [height, width])
    return image


def tf_read_image(file, width, height):
    image_string = tf.read_file(file)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image=tf_resize_image(image, width, height)
    image = tf.cast(image, tf.float32) * (1. / 255.0)  # 归一化
    return image

def map_read_image(files_list, labels_list):
    tf_image=tf_read_image(files_list,resize_width,resize_height)
    return tf_image,labels_list

def input_fun(files_list, labels_list, batch_size, shuffle=True):
    '''
    :param orig_image:
    :param dest_image:
    :param batch_size:
    :param num_epoch:
    :param shuffle:
    :return:
    '''
    # 构建数据集
    dataset = tf.data.Dataset.from_tensor_slices((files_list, labels_list))#TF version>=1.4
    # dataset = tf.contrib.data.Dataset.from_tensor_slices((files_list, labels_list))#TF version<1.4

    if shuffle:
        dataset = dataset.shuffle(100)
    dataset = dataset.repeat()  # 空为无限循环
    # dataset = dataset.map(map_read_image, num_parallel_calls=4)  # num_parallel_calls一般设置为cpu内核数量
    dataset = dataset.map(map_read_image)  # num_parallel_calls一般设置为cpu内核数量

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)  # software pipelining 机制
    dataset_iterator = dataset.make_one_shot_iterator()
    return dataset_iterator


def get_image_data(images_list, image_dir,labels_list, batch_size, re_height, re_width, shuffle=False):
    global resize_height
    global resize_width
    resize_height = re_height  # 指定存储图片高度
    resize_width = re_width    # 指定存储图片宽度
    image_list = [os.path.join(image_dir, name) for name in images_list]
    dataset = input_fun(image_list, labels_list, batch_size, shuffle)
    return dataset


def multilabel2onehot(batch_label, captcha_size, char_set_len):
    '''
    多标签转one-hot编码
    :param batch_label:
    :param captcha_size:
    :param char_set_len:
    :return:
    '''
    batch_size=batch_label.shape[0]
    vector = np.zeros(shape=(batch_size,captcha_size,char_set_len))
    for i,label in enumerate(batch_label):
        label = label.reshape(-1, 1)
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False, n_values=char_set_len)
        # onehot_encoder = preprocessing.OneHotEncoder(sparse=False, n_values='auto')
        # onehot_encoder = preprocessing.OneHotEncoder(sparse=False, categories=[range(char_set_len)])
        # onehot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')
        label_one_hot = onehot_encoder.fit_transform(label)
        vector[i,:,:]=label_one_hot
    vector=np.reshape(vector,newshape=(batch_size,-1))
    return vector


if __name__ == '__main__':
    data_filename='../dataset/test.txt'
    image_dir="../dataset/test"
    label_filename='../dataset/label_char_set.txt'
    char_set=file_processing.read_data(label_filename)

    images_list, labels_list=load_image_labels(data_filename)
    batch_size = 2
    captcha_size=4
    char_set_len=len(char_set)
    dataset=get_image_data(images_list, image_dir,labels_list, batch_size, re_height=None, re_width=None, shuffle=False)
    # 需满足：max_iterate*batch_size <=num_sample*num_epoch，否则越界
    max_iterate = 3
    with tf.Session() as sess:
        # dataset = dataset.make_initializable_iterator()
        # init_op = dataset.make_initializer(dataset)
        # sess.run(init_op)
        dataset = dataset.get_next()
        for i in range(max_iterate):
            batch_images, batch_label = sess.run(dataset)
            label_vector=multilabel2onehot(batch_label, captcha_size, char_set_len)

            label_name=file_processing.label_decode(batch_label[0,:],char_set)
            print('shape:{},tpye:{},labels:{}'.format(batch_images.shape, batch_images.dtype, batch_label))
            print("label_vector:{}".format(label_vector))
            show_image("image", batch_images[0, :, :, :])
            print("label name:{}".format(label_name))

