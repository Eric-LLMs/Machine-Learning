#!/usr/bin/python
#-*- encoding:utf-8 -*-
import math
import numpy as np
from LevelReading.levelreading_book_quan.conf.params import *

config = create_params_processing()
def Cosine(X, Y):
    vec_X, vec_Y = np.array(X), np.array(Y)
    return vec_X.dot(vec_Y)/(math.sqrt((vec_X**2).sum()) * math.sqrt((vec_Y**2).sum()))

def Cosine_list(X, Y_list):
    res = []
    for Y in Y_list:
        res.append(Cosine(X,Y))
    res_a = np.array(res)
    return res_a

if __name__=='__main__':
    pass