# -*-coding: utf-8 -*-
"""
    @Project: IntelligentManufacture
    @File   : image_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:34:50
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(title, image):
    '''
    调用matplotlib显示RGB图片
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


def cv_show_image(title, image):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :return:
    '''
    channels = image.shape[-1]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow(title, image)
    cv2.waitKey(0)


def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的RGB图片数据
    '''

    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    rgb_image = resize_image(rgb_image, resize_height, resize_width)
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image


def resize_image(image, resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape = np.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    if (resize_height is None) and (resize_width is None):  # 错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height = int(height * resize_width / width)
    elif resize_width is None:
        resize_width = int(width * resize_height / height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image


def scale_image(image, scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image, dsize=None, fx=scale[0], fy=scale[1])
    return image


def get_rect_image(image, rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h = rect
    cut_img = image[y:(y + h), x:(x + w)]
    return cut_img


def scale_rect(orig_rect, orig_shape, dest_shape):
    '''
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    '''
    new_x = int(orig_rect[0] * dest_shape[1] / orig_shape[1])
    new_y = int(orig_rect[1] * dest_shape[0] / orig_shape[0])
    new_w = int(orig_rect[2] * dest_shape[1] / orig_shape[1])
    new_h = int(orig_rect[3] * dest_shape[0] / orig_shape[0])
    dest_rect = [new_x, new_y, new_w, new_h]
    return dest_rect


def show_image_rect(win_name, image, rect):
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h = rect
    point1 = (x, y)
    point2 = (x + w, y + h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)


def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def save_image(image_path, rgb_image, toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)


def combime_save_image(orig_image, dest_image, out_dir, name, prefix):
    '''
    命名标准：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_" + prefix + ".jpg")
    save_image(dest_path, dest_image)

    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name, prefix)), dest_image)


if __name__ == "__main__":
    image_path = "../dataset/test_images/src.jpg"
    image = read_image(image_path, resize_height=None, resize_width=None)
    image = rgb_to_gray(image)
    orig_shape = np.shape(image)  # shape=(h,w)
    orig_rect = [50, 100, 100, 200]  # x,y,w,h
    print("orig_shape:{}".format(orig_shape))
    show_image_rect("orig", image, orig_rect)

    dest_image = resize_image(image, resize_height=None, resize_width=200)
    dest_shape = np.shape(dest_image)
    print("dest_shape:{}".format(dest_shape))
    dest_rect = scale_rect(orig_rect, orig_shape, dest_shape)
    show_image_rect("dest", dest_image, dest_rect)
