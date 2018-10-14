# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     fnns
   Description :
   Author :       huangneng
   date：          2018/8/7
-------------------------------------------------
   Change Activity:
                   2018/8/7:
-------------------------------------------------
"""
__author__ = 'huangneng'

import tensorflow as tf

def Fully_connected(x, out_num, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, units=out_num, use_bias=False)