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
from tensorflow.contrib.layers import batch_norm
import numpy as np
from baseunits.rnns import rnn_layers
from baseunits.fnns import Fully_connected
from tensorflow.contrib import framework
from baseunits.inception import create_inception_resnet_v1

vocab_size = 1024
embedding_size = 128


def Batch_Normalization(x, is_training, scope):
    with framework.arg_scope([batch_norm], scope=scope, updates_collections=None, decay=0.9, center=True, scale=True,
                             zero_debias_moving_mean=True):
        return tf.cond(is_training, lambda: batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=is_training, reuse=True))


def inception_layer(indata, training, times=16):
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1_maxpooling'):
        max_pool = tf.layers.max_pooling2d(
            indata, [1, 3], strides=1, padding="SAME", name="maxpool0a_1x3")
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
    with tf.variable_scope('branch5_residual_1x3'):
        conv_stem = tf.layers.conv2d(inputs=indata, filters=times * 3, kernel_size=[1, 1], strides=1, padding="SAME",
                                     use_bias=False, name='convstem_1x1')
        conv_stem = Batch_Normalization(
            conv_stem, is_training=training, scope='bn0')
        conv0e = tf.layers.conv2d(inputs=indata, filters=times * 2, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name='conv0e_1x1')
        conv0e = Batch_Normalization(conv0e, is_training=training, scope='bn1')
        conv0e = tf.nn.relu(conv0e)
        conv1e = tf.layers.conv2d(inputs=conv0e, filters=times*4, kernel_size=[1, 3], strides=1, padding="SAME",
                                  use_bias=False, name='conv1e_1x3')
        conv1e = Batch_Normalization(conv1e, is_training=training, scope='bn2')
        conv1e = tf.nn.relu(conv1e)
        conv2e = tf.layers.conv2d(inputs=conv1e, filters=times * 3, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name='conv2e_1x1')
        conv2e = Batch_Normalization(conv2e, is_training=training, scope='bn3')

        conv_plus = tf.add(conv_stem, conv2e)
        conv_plus = tf.nn.relu(conv_plus)
    return (tf.concat([conv1a, conv0b, conv1c, conv1d, conv_plus], axis=-1, name='concat'))


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
            rnn_out, rnn_states = rnn_layers(input_event, tf.ones_like(self.labels) * base_num, hidden_num=256,
                                             layer_num=3,
                                             kprob=self.keep_prob)
            self.fw_out = rnn_out[0]
            self.bw_out = rnn_out[1]
            extract_rnn_out = tf.concat(
                [rnn_out[0][:, -1, :], rnn_out[1][:, 0, :]], axis=1)  # [batch,2*hidden_num]
            extract_rnn_shape = extract_rnn_out.get_shape().as_list()
            event_model_output = extract_rnn_out

        with tf.name_scope("Signal_model"):
            input_signal = signals

            ############################################################################################################
            # x = tf.layers.conv2d(inputs=input_signal, filters=16, kernel_size=[1, 3], strides=1, padding="SAME",
            #                      use_bias=False,
            #                      name='conv1')
            # x = Batch_Normalization(x, is_training=self.training, scope='bn1')
            # x = tf.nn.relu(x)
            # x = tf.layers.max_pooling2d(x, [1, 2], strides=2, padding="SAME", name='maxpool1')
            #
            # x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[1, 3], strides=1, padding="SAME", use_bias=False,
            #                      name='conv2')
            # x = Batch_Normalization(x, is_training=self.training, scope='bn2')
            # x = tf.nn.relu(x)
            # x = tf.layers.max_pooling2d(x, [1, 2], strides=2, padding="SAME", name='maxpool2')
            #
            # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[1, 3], strides=1, padding="SAME", use_bias=False,
            #                      name='conv3')
            # x = Batch_Normalization(x, is_training=self.training, scope='bn3')
            # x = tf.nn.relu(x)
            # x = tf.layers.max_pooling2d(x, [1, 2], strides=2, padding="SAME", name='maxpool3')
            #
            # x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[1, 3], strides=1, padding="SAME", use_bias=False,
            #                      name='conv4')
            # x = Batch_Normalization(x, is_training=self.training, scope='bn4')
            # x = tf.nn.relu(x)
            # x = tf.layers.max_pooling2d(x, [1, 2], strides=2, padding="SAME", name='maxpool4')
            #
            # x_shape = x.get_shape().as_list()
            # signal_model_output = tf.reshape(x, [-1, x_shape[2] * x_shape[3]])  # [batch,1024]
            ############################################################################################################
            with tf.variable_scope("conv_layer1"):
                x = tf.layers.conv2d(inputs=input_signal, filters=64, kernel_size=[1, 7], strides=2, padding="SAME",
                                     use_bias=False, name="conv")
                x = Batch_Normalization(
                    x, is_training=self.training, scope='bn')
                x = tf.nn.relu(x)  # [188,64]
            with tf.variable_scope("maxpool_layer1"):
                x = tf.layers.max_pooling2d(
                    x, [1, 3], strides=2, padding="SAME", name="maxpool")  # [94,64]
            with tf.variable_scope("conv_layer2"):
                x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[1, 1], strides=1, padding="SAME",
                                     use_bias=False, name="conv")
                x = Batch_Normalization(
                    x, is_training=self.training, scope='bn')
                x = tf.nn.relu(x)  # [94,128]
            with tf.variable_scope("conv_layer3"):
                x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[1, 3], strides=1, padding="SAME",
                                     use_bias=False, name="conv")
                x = Batch_Normalization(
                    x, is_training=self.training, scope='bn')
                x = tf.nn.relu(x)  # [94,256]
            # inception layer x 11
            with tf.variable_scope('incp_layer1'):
                x = inception_layer(x, self.training)  # [94,192]
            with tf.variable_scope('incp_layer2'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer3'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('maxpool_layer2'):
                x = tf.layers.max_pooling2d(
                    x, [1, 3], strides=2, padding="SAME", name="maxpool")   # [47,192]
            with tf.variable_scope('incp_layer4'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer5'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer6'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer7'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer8'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('maxpool_layer3'):
                x = tf.layers.max_pooling2d(
                    x, [1, 3], strides=2, padding="SAME", name="maxpool")   # [24,192]
            with tf.variable_scope('incp_layer9'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer10'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('incp_layer11'):
                x = inception_layer(x, self.training)
            with tf.variable_scope('avgpool_layer1'):
                x = tf.layers.average_pooling2d(
                    x, [1, 7], strides=1, padding="SAME", name="avgpool")   # [24,192]
            print('inception output shape:', x.get_shape().as_list())
            x_shape = x.get_shape().as_list()
            signal_model_output = tf.reshape(x, [-1, x_shape[2] * x_shape[3]])
            ############################################################################################################
            # x = create_inception_resnet_v1(indata=input_signal,training=self.training,keep_prob=self.keep_prob)
            # print('inception output shape:', x.get_shape().as_list())
            # x_shape = x.get_shape().as_list()
            # signal_model_output = tf.reshape(x, [-1, x_shape[2] * x_shape[3]])

            ############################################################################################################
            # x = residual_cnn(indata=input_signal, out_channel=64, training=self.is_training, k=7, stride=2, scope="res_cnn_1")  # [N,188,64]
            # x = tf.layers.max_pooling2d(x, pool_size=[1, 3], strides=2, padding="SAME", name='mxp_1')  # [N,94,64]
            # x = residual_cnn(indata=x, out_channel=64, training=self.is_training, k=3, stride=1, scope="res_cnn_2")  # [N,94,64]
            #
            # x = residual_cnn(indata=x, out_channel=128, training=self.is_training, k=3, stride=2, scope="res_cnn_3")  # [N,47,128]
            # x = residual_cnn(indata=x, out_channel=128, training=self.is_training, k=3, stride=1, scope="res_cnn_4")  # [N,47,128]
            #
            # x = residual_cnn(indata=x, out_channel=256, training=self.is_training, k=3, stride=2, scope="res_cnn_5")  # [N,24,256]
            # x = residual_cnn(indata=x, out_channel=256, training=self.is_training, k=3, stride=1, scope="res_cnn_6")  # [N,24,256]
            # x_shape = x.get_shape().as_list()
            # signal_model_output = tf.reshape(x, [x_shape[0], x_shape[2] * x_shape[3]])  # [batch,6144]

            ############################################################################################################

            # x = residual_cnn(indata=input_signal, out_channel=16, training=self.training, k=7, stride=2,
            #                  scope="res_cnn_1")  # [N,188,64]
            # x = tf.layers.max_pooling2d(x, pool_size=[1, 3], strides=2, padding="SAME", name='mxp_1')  # [N,94,64]
            # x = residual_cnn(indata=x, out_channel=16, training=self.training, k=3, stride=1,
            #                  scope="res_cnn_2")  # [N,94,64]

            # x = residual_cnn(indata=x, out_channel=32, training=self.training, k=3, stride=2,
            #                  scope="res_cnn_3")  # [N,47,128]
            # x = residual_cnn(indata=x, out_channel=32, training=self.training, k=3, stride=1,
            #                  scope="res_cnn_4")  # [N,47,128]

            # x = residual_cnn(indata=x, out_channel=64, training=self.training, k=3, stride=2,
            #                  scope="res_cnn_5")  # [N,24,256]
            # x = residual_cnn(indata=x, out_channel=64, training=self.training, k=3, stride=1,
            #                  scope="res_cnn_6")  # [N,24,256]

            # x = residual_cnn(indata=x, out_channel=128, training=self.training, k=3, stride=2,
            #                  scope="res_cnn_7")  # [N,12,512]
            # x = residual_cnn(indata=x, out_channel=128, training=self.training, k=3, stride=1,
            #                  scope="res_cnn_8")  # [N,12,512]

            # x = residual_cnn(indata=x, out_channel=256, training=self.training, k=3, stride=2,
            #                  scope="res_cnn_9")  # [N,6,1024]
            # x = residual_cnn(indata=x, out_channel=256, training=self.training, k=3, stride=1,
            #                  scope="res_cnn_10")  # [N,6,1024]
            # x_shape = x.get_shape().as_list()
            # signal_model_output = tf.reshape(x, [x_shape[0], x_shape[2] * x_shape[3]])  # [batch,6*256]

        with tf.name_scope("Joint_model"):
            joint_input = tf.concat(
                [event_model_output, signal_model_output], axis=1)  # [batch,1536+256*2]
            joint_input_shape = joint_input.get_shape().as_list()
            fc1 = Fully_connected(
                joint_input, out_num=joint_input_shape[1], layer_name='joint_model_fc1')
            drop1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
            fc2 = Fully_connected(
                drop1, class_num, layer_name="joint_model_fc2")
            drop2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
            logits = drop2
        with tf.name_scope("train_opts"):
            self.activation_logits = tf.nn.sigmoid(logits)
            self.cost = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(self.cost)
            self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                    global_step=self.global_step)
            self.prediction = tf.argmax(self.activation_logits, axis=1)
            # self.accuracy = tf.reduce_mean(
            #     tf.cast(tf.equal(self.prediction, tf.argmax(one_hot_labels, axis=1)), dtype=tf.float32))
            # TODO : precision and recall
            # self.accuracy, self.accuracy_op = tf.metrics.accuracy(
            #     labels=self.labels, predictions=self.prediction, name="valid_metrics")
            # self.precision, self.precision_op = tf.metrics.precision(
            #     labels=self.labels, predictions=self.prediction, name="valid_metrics")
            # self.recall, self.recall_op = tf.metrics.recall(
            #     labels=self.labels, predictions=self.prediction, name="valid_metrics")
            # self.auc, self.auc_op = tf.metrics.auc(
            #     labels=self.labels, predictions=self.prediction, name="valid_metrics")
            # validation_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
            # self.running_validation_vars_init = tf.variables_initializer(
            #     var_list=validation_vars)
            # tf.summary.scalar('loss', self.loss)
            # tf.summary.scalar('accuarcy', self.accuracy)
            # tf.summary.scalar('recall', self.recall)
            # tf.summary.scalar('precision', self.precision)
            # tf.summary.scalar('auc', self.auc)
