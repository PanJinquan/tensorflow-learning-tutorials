# -*-coding: utf-8 -*-
"""
    @Project: tensorflow-learning-tutorials
    @File   : statistic.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-17 16:02:36
"""
import random
import os
import  numpy as np
from numpy import *;
# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    scores=scores/np.sum(scores,dtype=float32)
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return [mean,std]
def get_score(data,type):
    '''
    获得分数函数
    :param data: 输入数据1*10
    :param type: maxlabel:以最大值作为分数,
                 mean:    以加权均值作为分数
                 distribution:概率分布作为分数
    :return: 返回分数->float
    '''
    if type=='maxlabel':    # 以最大值作为分数
        max_index = data.index(max(data))
        score = round(max_index)
    elif type=='mean_score':# 平均分
        w=mat([1,2,3,4,5,6,7,8,9,10])
        sum_data=np.sum(data)
        pro_data=data/(sum_data*1.0)
        score=multiply(pro_data,w)
        score=np.sum(score)
        score=[score]
    elif type=='mean_pro':  # 平均分,归一化到[0,1]
        w=mat([1,2,3,4,5,6,7,8,9,10])
        sum_data=np.sum(data)
        pro_data=data/(sum_data*1.0)
        score=multiply(pro_data,w)
        score=np.sum(score)/10#归一化到[0,1]
        score=[score]
    elif type=='distribution':#概率分布作为分数
        score= get_distribution_score(data)
        score=score.tolist()
    elif type=='mean_std':# 均值和方差
        score=std_score(data)

    return score


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

###############################
def get_distribution_score(data,type=None):
    '''
    获得评分分布函数
    :param data:输入数据1*10
    :param type:None
    :return:
    '''
    sum_data=np.sum(data)
    pro_data=data/(sum_data*1.0)
    return pro_data
###############################
def labels_transform(labes,Omin,Omax):
    labes=np.asarray(labes)
    Imin=np.min(labes)
    Imax=np.max(labes)
    y = (Omax - Omin) * (labes - Imin) / (Imax - Imin) + Omin
    return y