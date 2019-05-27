# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : file_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-03 18:24:08
"""

def label_decode(label_list_index,label_set):
    '''
    label解码
    :return:
    '''
    label_list_name=[]
    for label_index in label_list_index:
        label_name=label_set[label_index]
        label_list_name.append(label_name)
    return label_list_name

def label_encode(label_list_name,label_set):
    '''
    label编码
    :return:
    '''
    label_list_index=[]
    for label_name in label_list_name:
        label_index = label_set.index(label_name)
        label_list_index.append(label_index)
    # label_list_index = label_set.index(label_list_name)
    return label_list_index


def write_data(file, content_list, model):
    with open(file, mode=model) as f:
        for line in content_list:
            f.write(line + "\n")


def read_data(file):
    with open(file, mode="r") as f:
        content_list = f.readlines()
        content_list = [content.rstrip() for content in content_list]
    return content_list





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

if __name__=='__main__':
    filename='../dataset/train.txt'
    images_list, labels_list=load_image_labels(filename)
    print(images_list)
    print(labels_list)