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
from __future__ import absolute_import

import numpy as np
from .layers import *

vocab_size = 1024
embedding_size = 128

logit_cf = 0.5


class Model(object):
    def __init__(self, base_num, signal_num, class_num, pos_weight=1.0,
                 is_cnn=True, is_base=True, is_rnn=True, model_prefix="model"):
        if (is_cnn | is_rnn) is False:
            raise ValueError("at least one of is_cnn/is_rnn should be True")
        with tf.name_scope(model_prefix + 'input'):
            self.base_int = tf.placeholder(tf.int32, [None, base_num])
            self.means = tf.placeholder(tf.float32, [None, base_num])
            self.stds = tf.placeholder(tf.float32, [None, base_num])
            self.sanums = tf.placeholder(tf.float32, [None, base_num])
            self.signals = tf.placeholder(
                tf.float32, [None, signal_num])  # middle base signals
            self.labels = tf.placeholder(tf.int32, [None])

        with tf.name_scope(model_prefix + 'input_params'):
            self.lr = tf.placeholder(tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)
            self.global_step = tf.get_variable(model_prefix + 'global_step', trainable=False, shape=(), dtype=tf.int32,
                                               initializer=tf.zeros_initializer())

        with tf.name_scope(model_prefix + 'models'):
            self.event_model = Event_model(layer_name=model_prefix + "em",
                                           sequence_len=tf.ones_like(self.labels) * base_num,
                                           cell="LSTM",
                                           layer_num=3,
                                           hidden_num=256,
                                           keep_prob=self.keep_prob)
            self.signal_model = incept_net(is_training=self.training, scopestr=model_prefix + "signalm")
            self.join_model = Joint_model(output_hidden=class_num,
                                          keep_prob=self.keep_prob)

        with tf.name_scope(model_prefix + "data_transfer"):
            one_hot_labels = tf.one_hot(self.labels, depth=class_num)
            if is_rnn:
                if is_base:
                    W = tf.get_variable(model_prefix + "embedding", shape=[vocab_size, embedding_size], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2. / vocab_size)))
                    embedded_base = tf.nn.embedding_lookup(W, self.base_int)
                    fusion_vector1 = tf.concat([embedded_base,
                                                tf.reshape(
                                                    self.means, [-1, base_num, 1]),
                                                tf.reshape(
                                                    self.stds, [-1, base_num, 1]),
                                                tf.reshape(self.sanums, [-1, base_num, 1])], axis=2)
                else:
                    fusion_vector1 = tf.concat([tf.reshape(
                                                    self.means, [-1, base_num, 1]),
                                                tf.reshape(
                                                    self.stds, [-1, base_num, 1]),
                                                tf.reshape(self.sanums, [-1, base_num, 1])], axis=2)
            signals = tf.reshape(self.signals, [-1, 1, signal_num, 1])

        with tf.name_scope(model_prefix + "Event_model"):
            if is_rnn:
                input_event = fusion_vector1
                event_model_output = self.event_model(input_event)

        with tf.name_scope(model_prefix + "Signal_model"):
            input_signal = signals
            signal_model_output = self.signal_model(input_signal)
            # signal_model_output = None

        with tf.name_scope(model_prefix + "Joint_model"):
            if is_cnn:
                if is_rnn:
                    logits = self.join_model(event_model_output, signal_model_output)
                else:
                    logits = self.join_model(None, signal_model_output)
            else:
                logits = self.join_model(event_model_output, None)
            logits1 = tf.cast(tf.squeeze(tf.slice(logits, [0, 1], [tf.shape(logits)[0], 1])),
                              tf.float32)
        with tf.name_scope(model_prefix + "train_opts"):
            # >>sigmoid performs well here, but softmax may be more appropriate<<
            self.activation_logits = tf.nn.sigmoid(logits)
            self.activation_logits1 = tf.cast(tf.squeeze(tf.slice(self.activation_logits, [0, 1],
                                                                  [tf.shape(self.activation_logits)[0], 1])),
                                              tf.float32)

            # >>need more tests to decide whether to use ONE_HOT in WEIGHTED_CROSS_ENTROPY or not<<
            # >>especially when pos_weight != 1<<
            if pos_weight == 1.0:
                self.prediction = tf.argmax(self.activation_logits, axis=1)
                self.loss_pw = tf.nn.weighted_cross_entropy_with_logits(
                    logits=logits, targets=one_hot_labels, pos_weight=pos_weight)
            else:
                self.prediction = tf.where(tf.greater(self.activation_logits1, 0.5),
                                           tf.ones_like(self.activation_logits1, dtype=tf.int32),
                                           tf.zeros_like(self.activation_logits1, dtype=tf.int32))
                self.loss_pw = tf.nn.weighted_cross_entropy_with_logits(
                    logits=logits1, targets=tf.cast(self.labels, tf.float32), pos_weight=pos_weight)

            self.cost_pw = tf.reduce_mean(self.loss_pw)
            self.train_opt_pw = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost_pw,
                                                                                       global_step=self.global_step)
