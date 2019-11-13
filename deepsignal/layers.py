"""
/*
 * @Author: huangneng@clemson
 * @Email: {huangneng@csu.edu.cn}
 * @Date: 2018-11-08 14:44:17
 * @Last Modified by: huangneng@clemson
 * @Last Modified time: 2018-11-08 14:57:11
*/
"""

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib import framework


def rnn_layers(x,
               seq_length,
               hidden_num=100,
               layer_num=3,
               cell='LSTM',
               kprob=0.8,
               layer_name="brnn",
               dtype=tf.float32):
    """Generate RNN layers.

    Args:
        x (Float): A 3D-Tensor of shape [batch_size,max_time,channel]
        seq_length (Int): A 1D-Tensor of shape [batch_size], real length of each sequence.
        training (Boolean): A 0D-Tenosr indicate if it's in training.
        hidden_num (int, optional): Defaults to 100. Size of the hidden state,
            hidden unit will be deep concatenated, so the final hidden state will be size of 200.
        layer_num (int, optional): Defaults to 3. Number of layers in RNN.
        class_n (int, optional): Defaults to 5. Number of output class.
        cell(str): A String from 'LSTM','GRU','BNLSTM', the RNN Cell used.
            BNLSTM stand for Batch normalization LSTM Cell.

    Returns:
         logits: A 3D Tensor of shape [batch_size, max_time, class_n]
    """

    cells_fw = list()
    cells_bw = list()
    for i in range(layer_num):
        if cell == 'LSTM':
            cell_fw = LSTMCell(hidden_num)
            cell_bw = LSTMCell(hidden_num)
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw, output_keep_prob=kprob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, output_keep_prob=kprob)
        elif cell == 'GRU':
            cell_fw = GRUCell(hidden_num)
            cell_bw = GRUCell(hidden_num)
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw, output_keep_prob=kprob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, output_keep_prob=kprob)
        else:
            raise ValueError("Cell type unrecognized.")
        cells_fw.append(cell_fw)
        cells_bw.append(cell_bw)
    multi_cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    multi_cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
    with tf.variable_scope(layer_name) as scope:
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_cells_fw, cell_bw=multi_cells_bw, inputs=x, sequence_length=seq_length, dtype=dtype,
            scope=scope)
    return outputs, states


def Fully_connected(x, out_num, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, units=out_num, use_bias=False)


def Batch_Normalization(x, is_training, scope):
    with framework.arg_scope([batch_norm], scope=scope, updates_collections=None, decay=0.9, center=True, scale=True,
                             zero_debias_moving_mean=True):
        return tf.cond(is_training, lambda: batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=is_training, reuse=True))


def inception_layer(indata, training, scope_str="inception_layer", times=16):
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope(scope_str + 'branch1_maxpooling'):
        max_pool = tf.layers.max_pooling2d(
            indata, [1, 3], strides=1, padding="SAME", name="maxpool0a_1x3")
        conv1a = tf.layers.conv2d(inputs=max_pool, filters=times * 3, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name='conv1a_1x1')
        conv1a = Batch_Normalization(conv1a, is_training=training, scope='bn')
        conv1a = tf.nn.relu(conv1a)
    with tf.variable_scope(scope_str + 'branch2_1x1'):
        conv0b = tf.layers.conv2d(inputs=indata, filters=times * 3, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name='conv0b_1x1')
        conv0b = Batch_Normalization(conv0b, is_training=training, scope='bn')
        conv0b = tf.nn.relu(conv0b)
    with tf.variable_scope(scope_str + 'branch3_1x3'):
        conv0c = tf.layers.conv2d(inputs=indata, filters=times * 2, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name="conv0c_1x1")
        conv0c = Batch_Normalization(conv0c, is_training=training, scope='bn1')
        conv0c = tf.nn.relu(conv0c)
        conv1c = tf.layers.conv2d(inputs=conv0c, filters=times * 3, kernel_size=[1, 3], strides=1, padding="SAME",
                                  use_bias=False, name="conv1c_1x3")
        conv1c = Batch_Normalization(conv1c, is_training=training, scope='bn2')
        conv1c = tf.nn.relu(conv1c)
    with tf.variable_scope(scope_str + 'branch4_1x5'):
        conv0d = tf.layers.conv2d(inputs=indata, filters=times * 2, kernel_size=[1, 1], strides=1, padding="SAME",
                                  use_bias=False, name="conv0d_1x1")
        conv0d = Batch_Normalization(conv0d, is_training=training, scope='bn1')
        conv0d = tf.nn.relu(conv0d)
        conv1d = tf.layers.conv2d(inputs=conv0d, filters=times * 3, kernel_size=[1, 5], strides=1, padding="SAME",
                                  use_bias=False, name="conv1d_1x5")
        conv1d = Batch_Normalization(conv1d, is_training=training, scope='bn2')
        conv1d = tf.nn.relu(conv1d)
    with tf.variable_scope(scope_str + 'branch5_residual_1x3'):
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


class Event_model():
    """
    A multi layer bidirectional recurent neural network
    """

    def __init__(self,
                 sequence_len,
                 cell,
                 layer_num,
                 hidden_num,
                 keep_prob,
                 layer_name="eventmodel"):
        self.seq_length = sequence_len
        self.cell = cell
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.keep_prob = keep_prob
        self.layer_name = layer_name

    def __call__(self, input):
        rnn_out, rnn_states = rnn_layers(input,
                                         cell=self.cell,
                                         seq_length=self.seq_length,
                                         hidden_num=self.hidden_num,
                                         layer_num=self.layer_num,
                                         kprob=self.keep_prob,
                                         layer_name=self.layer_name)
        fw_out = rnn_out[0]
        bw_out = rnn_out[1]
        extract_rnn_out = tf.concat(
            [rnn_out[0][:, -1, :], rnn_out[1][:, 0, :]], axis=1)
        return extract_rnn_out


class incept_net():
    def __init__(self, is_training, scopestr="inception_net"):
        self.training = is_training
        self.scopestr = scopestr

    def __call__(self, signals):
        input_signal = signals
        with tf.variable_scope(self.scopestr + "conv_layer1"):
            x = tf.layers.conv2d(inputs=input_signal, filters=64, kernel_size=[1, 7], strides=2, padding="SAME",
                                 use_bias=False, name="conv")
            x = Batch_Normalization(
                x, is_training=self.training, scope='bn')
            x = tf.nn.relu(x)  # [188,64]
        with tf.variable_scope(self.scopestr + "maxpool_layer1"):
            x = tf.layers.max_pooling2d(
                x, [1, 3], strides=2, padding="SAME", name="maxpool")  # [94,64]
        with tf.variable_scope(self.scopestr + "conv_layer2"):
            x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[1, 1], strides=1, padding="SAME",
                                 use_bias=False, name="conv")
            x = Batch_Normalization(
                x, is_training=self.training, scope='bn')
            x = tf.nn.relu(x)  # [94,128]
        with tf.variable_scope(self.scopestr + "conv_layer3"):
            x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[1, 3], strides=1, padding="SAME",
                                 use_bias=False, name="conv")
            x = Batch_Normalization(
                x, is_training=self.training, scope='bn')
            x = tf.nn.relu(x)  # [94,256]
        # inception layer x 11
        with tf.variable_scope(self.scopestr + 'incp_layer1'):
            x = inception_layer(x, self.training, self.scopestr + "1")  # [94,192]
        with tf.variable_scope(self.scopestr + 'incp_layer2'):
            x = inception_layer(x, self.training, self.scopestr + "2")
        with tf.variable_scope(self.scopestr + 'incp_layer3'):
            x = inception_layer(x, self.training, self.scopestr + "3")
        with tf.variable_scope(self.scopestr + 'maxpool_layer2'):
            x = tf.layers.max_pooling2d(
                x, [1, 3], strides=2, padding="SAME", name="maxpool")   # [47,192]
        with tf.variable_scope(self.scopestr + 'incp_layer4'):
            x = inception_layer(x, self.training, self.scopestr + "4")
        with tf.variable_scope(self.scopestr + 'incp_layer5'):
            x = inception_layer(x, self.training, self.scopestr + "5")
        with tf.variable_scope(self.scopestr + 'incp_layer6'):
            x = inception_layer(x, self.training, self.scopestr + "6")
        with tf.variable_scope(self.scopestr + 'incp_layer7'):
            x = inception_layer(x, self.training, self.scopestr + "7")
        with tf.variable_scope(self.scopestr + 'incp_layer8'):
            x = inception_layer(x, self.training, self.scopestr + "8")
        with tf.variable_scope(self.scopestr + 'maxpool_layer3'):
            x = tf.layers.max_pooling2d(
                x, [1, 3], strides=2, padding="SAME", name="maxpool")   # [24,192]
        with tf.variable_scope(self.scopestr + 'incp_layer9'):
            x = inception_layer(x, self.training, self.scopestr + "9")
        with tf.variable_scope(self.scopestr + 'incp_layer10'):
            x = inception_layer(x, self.training, self.scopestr + "10")
        with tf.variable_scope(self.scopestr + 'incp_layer11'):
            x = inception_layer(x, self.training, self.scopestr + "11")
        with tf.variable_scope(self.scopestr + 'avgpool_layer1'):
            x = tf.layers.average_pooling2d(
                x, [1, 7], strides=1, padding="SAME", name="avgpool")   # [24,192]
        # print('inception output shape:', x.get_shape().as_list())
        x_shape = x.get_shape().as_list()
        signal_model_output = tf.reshape(x, [-1, x_shape[2] * x_shape[3]])
        return signal_model_output


class Joint_model():
    def __init__(self, output_hidden, keep_prob):
        self.output_hidden = output_hidden
        self.keep_prob = keep_prob

    def __call__(self, event_model_output, signal_model_output):
        if signal_model_output is not None:
            if event_model_output is not None:
                joint_input = tf.concat(
                    [event_model_output, signal_model_output], axis=1)  # [batch,1536+256*2]
            else:
                joint_input = signal_model_output
        else:
            joint_input = event_model_output

        joint_input_shape = joint_input.get_shape().as_list()
        fc1 = Fully_connected(
            joint_input, out_num=joint_input_shape[1], layer_name='joint_model_fc1')
        drop1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob)
        fc2 = Fully_connected(
            drop1, self.output_hidden, layer_name="joint_model_fc2")
        drop2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
        return drop2
