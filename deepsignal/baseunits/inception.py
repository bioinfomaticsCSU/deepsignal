#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: 
# @Date: 2018-08-16 15:31
# @author: huangneng
# @contact: huangneng@csu.edu.cn

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, conv2d, max_pool2d, avg_pool2d, dropout
from tensorflow.contrib import framework


def Batch_Normalization(x, is_training, scope):
    with framework.arg_scope([batch_norm], scope=scope, updates_collections=None, decay=0.9, center=True, scale=True,
                             zero_debias_moving_mean=True):
        return tf.cond(is_training, lambda: batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=is_training, reuse=True))


def inception_layer(indata, training, times=16):
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1_maxpooling'):
        max_pool = tf.layers.max_pooling2d(indata, [1, 3], strides=1, padding="SAME", name="maxpool0a_1x3")
        conv1a = tf.layers.conv2d(inputs=max_pool, filters=times * 3, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name='conv1a_1x1')
        conv1a = Batch_Normalization(conv1a, is_training=training, scope='bn')
        conv1a = tf.nn.relu(conv1a)
    with tf.variable_scope('branch2_1x1'):
        conv0b = tf.layers.conv2d(inputs=indata, filters=times * 3, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name='conv0b_1x1')
        conv0b = Batch_Normalization(conv0b, is_training=training, scope='bn')
        conv0b = tf.nn.relu(conv0b)
    with tf.variable_scope('branch3_1x3'):
        conv0c = tf.layers.conv2d(inputs=indata, filters=times * 2, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name="conv0c_1x1")
        conv0c = Batch_Normalization(conv0c, is_training=training, scope='bn1')
        conv0c = tf.nn.relu(conv0c)
        conv1c = tf.layers.conv2d(inputs=conv0c, filters=times * 3, kernel_size=[1, 3], strides=1, padding="SAME",
                                  use_bias=False, name="conv1c_1x3")
        conv1c = Batch_Normalization(conv1c, is_training=training, scope='bn2')
        conv1c = tf.nn.relu(conv1c)
    with tf.variable_scope('branch4_1x5'):
        conv0d = tf.layers.conv2d(inputs=indata, filters=times * 2, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name="conv0d_1x1")
        conv0d = Batch_Normalization(conv0d, is_training=training, scope='bn1')
        conv0d = tf.nn.relu(conv0d)
        conv1d = tf.layers.conv2d(inputs=conv0d, filters=times * 3, kernel_size=[1, 5], strides=1, padding="SAME",
                                  use_bias=False, name="conv1d_1x5")
        conv1d = Batch_Normalization(conv1d, is_training=training, scope='bn2')
        conv1d = tf.nn.relu(conv1d)
    return (tf.concat([conv1a, conv0b, conv1c, conv1d], axis=-1, name='concat'))


def stem(indata):
    with tf.variable_scope('conv_layer1'):
        x = conv2d(inputs=indata, num_outputs=32, kernel_size=[1, 3], stride=2, padding="VALID")
    with tf.variable_scope('conv_layer2'):
        x = conv2d(inputs=x, num_outputs=32, kernel_size=[1, 3], padding="VALID")
    with tf.variable_scope('conv_layer3'):
        x = conv2d(inputs=x, num_outputs=64, kernel_size=[1, 3], padding="SAME")
    with tf.variable_scope('maxpool_layer1'):
        x = max_pool2d(inputs=x, kernel_size=[1, 3], stride=2, padding="VALID")
    with tf.variable_scope('conv_layer4'):
        x = conv2d(inputs=x, num_outputs=80, kernel_size=[1, 1], padding="SAME")
    with tf.variable_scope('conv_layer5'):
        x = conv2d(inputs=x, num_outputs=192, kernel_size=[1, 3], padding="VALID")
    with tf.variable_scope('conv_layer6'):
        x = conv2d(inputs=x, num_outputs=256, kernel_size=[1, 3], stride=2, padding="VALID")
    return x


def Inception_A_ResNet_v1(indata, training, scale_residual=True):
    init = indata
    ir1 = conv2d(inputs=indata, num_outputs=32, kernel_size=[1, 1], padding="SAME", scope='branch_a_0')

    ir2 = conv2d(inputs=indata, num_outputs=32, kernel_size=[1, 1], padding="SAME", scope='branch_b_0')
    ir2 = conv2d(inputs=ir2, num_outputs=32, kernel_size=[1, 3], padding="SAME", scope='branch_b_1')

    ir3 = conv2d(inputs=indata, num_outputs=32, kernel_size=[1, 1], padding="SAME", scope='branch_c_0')
    ir3 = conv2d(inputs=ir3, num_outputs=32, kernel_size=[1, 3], padding="SAME", scope='branch_c_1')
    ir3 = conv2d(inputs=ir3, num_outputs=32, kernel_size=[1, 3], padding="SAME", scope='branch_c_2')

    mer1 = tf.concat([ir1, ir2, ir3], axis=-1, name='merge1')

    ir_conv = conv2d(inputs=mer1, num_outputs=256, kernel_size=[1, 1], activation_fn=None, padding="SAME",
                     scope='ir_conv')

    if scale_residual: ir_conv = ir_conv * 0.1

    out = init+ir_conv
    out = Batch_Normalization(out, training, scope='bn')
    out = tf.nn.relu(out)
    return out


def Inception_B_ResNet_v1(indata, training, scale_residual=True):
    init = indata
    ir1 = conv2d(inputs=indata, num_outputs=128, kernel_size=[1, 1], padding="SAME", scope='branch_a_0')

    ir2 = conv2d(inputs=indata, num_outputs=128, kernel_size=[1, 1], padding="SAME", scope='branch_b_0')
    ir2 = conv2d(inputs=ir2, num_outputs=128, kernel_size=[1, 7], padding="SAME", scope='branch_b_1')

    mer1 = tf.concat([ir1, ir2], axis=-1, name='merge1')

    ir_conv = conv2d(inputs=mer1, num_outputs=896, kernel_size=[1, 1], activation_fn=None, padding="SAME",
                     scope='ir_conv')

    if scale_residual: ir_conv = ir_conv * 0.1

    out = init + ir_conv
    out = Batch_Normalization(out, training, scope='bn')
    out = tf.nn.relu(out)
    return out


def Inception_C_ResNet_v1(indata, training, scale_residual=True):
    init = indata
    ir1 = conv2d(inputs=indata, num_outputs=192, kernel_size=[1, 1], padding="SAME", scope='branch_a_0')

    ir2 = conv2d(inputs=indata, num_outputs=192, kernel_size=[1, 1], padding="SAME", scope='branch_b_0')
    ir2 = conv2d(inputs=ir2, num_outputs=192, kernel_size=[1, 3], padding="SAME", scope='branch_b_1')

    mer1 = tf.concat([ir1, ir2], axis=-1, name='merge1')

    ir_conv = conv2d(inputs=mer1, num_outputs=1792, kernel_size=[1, 1], activation_fn=None, padding="SAME",
                     scope='ir_conv')

    if scale_residual: ir_conv = ir_conv * 0.1

    out = init + ir_conv
    out = Batch_Normalization(out, training, scope='bn')
    out = tf.nn.relu(out)
    return out


def reduction_A(indata, training, k=192, l=224, m=256, n=384):
    r1 = max_pool2d(inputs=indata, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_a_0')

    r2 = conv2d(inputs=indata, num_outputs=n, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_b_0')

    r3 = conv2d(inputs=indata, num_outputs=k, kernel_size=[1, 1], padding="SAME", scope='branch_c_0')
    r3 = conv2d(inputs=r3, num_outputs=l, kernel_size=[1, 3], padding="SAME", scope='branch_c_1')
    r3 = conv2d(inputs=r3, num_outputs=m, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_c_2')

    out = tf.concat([r1, r2, r3], axis=-1, name='merge1')
    out = Batch_Normalization(out, training, scope='bn1')
    out = tf.nn.relu(out)
    return out


def reduction_B(indata, training):
    r1 = max_pool2d(inputs=indata, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_a_0')

    r2 = conv2d(inputs=indata, num_outputs=256, kernel_size=[1, 1], padding="SAME", scope='branch_b_0')
    r2 = conv2d(inputs=r2, num_outputs=384, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_b_1')

    r3 = conv2d(inputs=indata, num_outputs=256, kernel_size=[1, 1], padding="SAME", scope='branch_c_0')
    r3 = conv2d(inputs=r3, num_outputs=256, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_c_1')

    r4 = conv2d(inputs=indata, num_outputs=256, kernel_size=[1, 1], padding="SAME", scope='branch_d_0')
    r4 = conv2d(inputs=r4, num_outputs=256, kernel_size=[1, 3], padding="SAME", scope='branch_d_1')
    r4 = conv2d(inputs=r4, num_outputs=256, kernel_size=[1, 3], stride=2, padding="VALID", scope='branch_d_2')

    out = tf.concat([r1, r2, r3, r4], axis=-1, name='merge1')
    out = Batch_Normalization(out, training, scope='bn1')
    out = tf.nn.relu(out)
    return out


def create_inception_resnet_v1(indata, training, keep_prob=0.8, scale=True):
    with tf.variable_scope('stem'):
        x = stem(indata)
        # print('output shape:', x.get_shape().as_list())

    # 5 x inception resnet A
    for i in range(5):
        with tf.variable_scope('inception_A_' + str(i)):
            x = Inception_A_ResNet_v1(x, training, scale)
            # print('output shape:', x.get_shape().as_list())

    # reduction A
    with tf.variable_scope('reduction_A'):
        x = reduction_A(x, training)
        # print('output shape:', x.get_shape().as_list())

    # 10 x inception resnet B
    for i in range(10):
        with tf.variable_scope('inception_B_' + str(i)):
            x = Inception_B_ResNet_v1(x, training, scale)
            # print('output shape:', x.get_shape().as_list())

    # reduction B
    with tf.variable_scope('reduction_B'):
        x = reduction_B(x, training)
        # print('output shape:', x.get_shape().as_list())

    # 5 x inception resnet C
    for i in range(5):
        with tf.variable_scope('inception_C_' + str(i)):
            x = Inception_C_ResNet_v1(x, training, scale)
            # print('output shape:', x.get_shape().as_list())

    # average pooling
    with tf.variable_scope('average_pool'):
        x = avg_pool2d(inputs=x, kernel_size=[1, 8], stride=1, padding="SAME")
        # print('output shape:', x.get_shape().as_list())

    # dropout
    with tf.variable_scope('dropout'):
        x = dropout(x, keep_prob)
        # print('output shape:', x.get_shape().as_list())

    # with tf.variable_scope('dense'):
    #     x_shape=x.get_shape().as_list()
    #     x = tf.reshape(x,[-1,x_shape[2]*x_shape[3]])
    #     x = dense(inputs=x,units=2048)
    return x
