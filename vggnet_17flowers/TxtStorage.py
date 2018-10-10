# -*-coding: utf-8 -*-
"""
 @Project: TxtStorage
 @File   : TxtStorage.py
 @Author : panjq
 @E-mail : pan_jinquan@163.com
 @Date   : 2018-07-12 17:32:47
"""
# -*- coding: utf-8 -*-
from numpy import *


class TxtStorage:
    # def __init__(self):

    # 保存txt数据
    def save_txt(self, content, filename, mode='a'):
        """保存txt数据
        :param content:需要保存的数据
        :param filename:文件名
        :param mode:读写模式
        :return: void
        """
        file = open(filename, mode)
        for row in range(len(content)):
            row_data = content[row]
            for col in range(len(row_data)):
                data = row_data[col]
                if not col == len(row_data) - 1:
                    file.write(str(data) + ' ')
                else:
                    file.write(str(data))
            file.write('\n')
        file.close()

    # 读取txt数据函数
    def read_txt(self, fileName):
        """读取txt数据函数
        :param filename:文件名
        :return: txt的数据列表
        :rtype: list
        """
        try:
            file = open(fileName, 'r')

        except IOError:
            print('read txt file data failed....')
            # error = []
            return None
        Data = []
        with file as txtData:
            lines = txtData.readlines()
            for line in lines:
                lineData = line.strip()  # 去除空白和逗号“,”
                Data.append(lineData)
        return Data

    # 按空格分割字符串，并以列表的形式返回
    def splitData(self, dataSet):
        """分割字符串
        :param dataSet:文件名
        :return: 按空格分割字符串，并以列表的形式返回
        :rtype: list
        """
        re = []
        for str in dataSet:
            str_list = str.split()
            int_list = []
            for i in str_list:
                if i.isdigit():
                    int_list.append(int(i))
                else:
                    int_list.append(i)
            re.append(int_list)
        return re


if __name__ == '__main__':
    txt_filename = 'test.txt'
    txt_data = [['1.jpg', 'dog', 200, 300], ['2.jpg', 'dog', 20, 30]]
    txt_str = TxtStorage()
    txt_str.save_txt(txt_data, txt_filename, mode='w')
    data = txt_str.read_txt(txt_filename)
    print(data)
    data = txt_str.splitData(data)
    for image_name, label, img_row, img_col in data:
        print(image_name, label, img_row, img_col)
