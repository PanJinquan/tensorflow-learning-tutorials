# -*-coding: utf-8 -*-
"""
    @Project: verification_code
    @File   : create_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-01 21:24:51
"""
import numpy as np
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
from utils import file_processing,image_processing
import os

def random_captcha_text(char_set, captcha_size):
    '''
    从字符集中随机选取captcha_size个字符
    :param char_set: 字符集
    :param captcha_size: 字符个数
    :return:
    '''
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(char_set, captcha_height, captcha_width, captcha_size):
    '''
    产生验证码
    :param char_set: 验证码字符集
    :param captcha_height: 验证码图片高度
    :param captcha_width: 验证码图片宽度
    :param captcha_size: 验证码含有字符个数
    :return:
    '''
    image = ImageCaptcha(width=captcha_width,height=captcha_height)
    # 随机产生验证码字符
    captcha_text = random_captcha_text(char_set=char_set, captcha_size=captcha_size)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text, captcha_size, char_set_len):
    '''
    文本转向量：字符编码
    :param text:
    :param captcha_size:
    :param char_set_len:
    :return:
    '''
    text_len = len(text)
    if text_len > captcha_size:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(captcha_size * char_set_len)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    # test_data = np.reshape(vector, (4, -1))
    return vector


def vec2text(vec, char_set_len):
    '''
    向量转回文本:字符解码
    :param vec:
    :param char_set_len:
    :return:
    '''
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % char_set_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)



def get_next_batch(char_set, shape, captcha_size):
    '''
    直接调用captcha生成一个batch训练数据
    :param char_set:
    :param shape: =[batch_size, image_height, image_width,depth]
    :param captcha_size:
    :param char_set_len:
    :return:
    '''
    char_set_len = len(char_set)#字符种类个数
    [batch_size, image_height, image_width, depth]=shape
    batch_x = np.zeros(shape=shape)
    batch_y = np.zeros([batch_size, captcha_size * char_set_len])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(char_set=char_set,
                                                     captcha_height=image_height,
                                                     captcha_width=image_width,
                                                     captcha_size=captcha_size)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        # image = convert2gray(image)

        # batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_x[i, :] = image / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text, captcha_size, char_set_len)
    return batch_x, batch_y



def create_dataset(out_dir,nums,filename,char_set, captcha_height,captcha_width,captcha_size):
    '''
    产生样本
    :param out_dir: 数据集图片保存目录
    :param nums: 产生数据样本个数
    :param filename: 保存数据txt文件
    :param char_set: 字符数据集
    :param captcha_height: 验证码height
    :param captcha_width:  验证码width
    :param captcha_size:   验证码大小
    :return:None
    '''

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # 产生一个验证码样本并显示
    i=0
    while i<nums:
        text, image = gen_captcha_text_and_image(char_set=char_set,
                                                 captcha_height=captcha_height,
                                                 captcha_width=captcha_width,
                                                 captcha_size=captcha_size)
        # 产生的验证码图并不一定等于image_height*image_width
        if image.shape != (image_height, image_width, 3):
            continue

        if i ==0:
            image_processing.cv_show_image(text, image)  # 显示验证码

        image_name=str(i)+"_"+text+".jpg"
        image_path=out_dir+"/"+image_name
        print(image_path)
        image_processing.save_image(image_path, image, toUINT8=False)
        text=[c for c in text]
        label_list=file_processing.label_encode(text,char_set)
        content=[image_name]+label_list
        content = ' '.join('%s' % id for id in content)
        file_processing.write_data(filename, [content], model='a')
        i+=1


def save_label_set(filename,label_name_set):
    '''
    保存label数据
    :param filename:
    :param label_name_set:
    :return:
    '''
    # 将字符集char_set转为整形的标签集合
    # label_set=list(range(0,len(label_name_set)))
    content_list=[]
    for label_index,name in enumerate(label_name_set):
        # content=name +" "+str(label_index)
        content=name
        content_list.append(content)
    file_processing.write_data(filename,content_list,model='w')

if __name__=="__main__":
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    '''    
    这里验证码字符集只考虑大写字母的情况,并且验证码大小captcha_size=4
    如果要预测大小写字母和数字，产生数据时，请加上number，alphabet和ALPHABET字符集
    '''
    # char_set = number + alphabet + ALPHABET
    # char_set = ALPHABET
    char_set = number

    # 保存字符集
    label_filename='../dataset/label_char_set.txt'
    # label_filename='../dataset/label_number.txt'
    save_label_set(label_filename,char_set)#保存字符集

    batch_size = 32
    image_height = 60
    image_width = 160
    depth = 3
    captcha_size = 4

    # 产生train数据
    out_dir="../dataset/train"
    train_nums=20000
    filename='../dataset/train.txt'
    create_dataset(out_dir,train_nums,filename,char_set,image_height,image_width,captcha_size)

    # 产生test数据
    out_dir="../dataset/test"
    test_nums=100
    filename='../dataset/test.txt'
    create_dataset(out_dir,test_nums,filename,char_set,image_height,image_width,captcha_size)