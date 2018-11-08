# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       huangneng
   date：          2018/8/6
-------------------------------------------------
   Change Activity:
                   2018/8/6:
-------------------------------------------------
"""
__author__ = 'huangneng'

import tensorflow as tf
import numpy as np
from layers import *

vocab_size = 1024
embedding_size = 128


class Model():
    def __init__(self, base_num, signal_num, class_num):
        with tf.name_scope('input'):
            self.base_int = tf.placeholder(tf.int32, [None, base_num])
            self.means = tf.placeholder(tf.float32, [None, base_num])
            self.stds = tf.placeholder(tf.float32, [None, base_num])
            self.sanums = tf.placeholder(tf.float32, [None, base_num])
            self.signals = tf.placeholder(
                tf.float32, [None, signal_num])  # middle base signals
            self.labels = tf.placeholder(tf.int32, [None])

        with tf.name_scope('input_params'):
            self.lr = tf.placeholder(tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)
            self.global_step = tf.get_variable('global_step', trainable=False, shape=(), dtype=tf.int32,
                                               initializer=tf.zeros_initializer())

        with tf.name_scope('models'):
            self.event_model = Event_model(sequence_len=tf.ones_like(self.labels) * base_num,
                                           cell="LSTM",
                                           layer_num=3,
                                           hidden_num=256,
                                           keep_prob=self.keep_prob)
            self.signal_model = incept_net(is_training=self.training)
            self.join_model = Joint_model(output_hidden=class_num,
                                          keep_prob=self.keep_prob)

        with tf.name_scope("data_transfer"):
            one_hot_labels = tf.one_hot(self.labels, depth=class_num)
            W = tf.get_variable("embedding", shape=[vocab_size, embedding_size], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2. / vocab_size)))
            embedded_base = tf.nn.embedding_lookup(W, self.base_int)
            fusion_vector1 = tf.concat([embedded_base,
                                        tf.reshape(
                                            self.means, [-1, base_num, 1]),
                                        tf.reshape(
                                            self.stds, [-1, base_num, 1]),
                                        tf.reshape(self.sanums, [-1, base_num, 1])], axis=2)
            signals = tf.reshape(self.signals, [-1, 1, signal_num, 1])

        with tf.name_scope("Event_model"):
            input_event = fusion_vector1
            event_model_output = self.event_model(input_event)

        with tf.name_scope("Signal_model"):
            input_signal = signals
            signal_model_output = self.signal_model(input_signal)
            # signal_model_output = None

        with tf.name_scope("Joint_model"):
            logits = self.join_model(event_model_output, signal_model_output)
        with tf.name_scope("train_opts"):
            self.activation_logits = tf.nn.sigmoid(logits)
            self.cost = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(self.cost)
            self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                    global_step=self.global_step)
            self.prediction = tf.argmax(self.activation_logits, axis=1)
