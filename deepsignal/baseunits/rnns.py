# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     rnns.py
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
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell


def rnn_layers(x,
               seq_length,
               hidden_num=100,
               layer_num=3,
               cell='LSTM',
               kprob=0.8,
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
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=kprob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=kprob)
        elif cell == 'GRU':
            cell_fw = GRUCell(hidden_num)
            cell_bw = GRUCell(hidden_num)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=kprob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=kprob)
        else:
            raise ValueError("Cell type unrecognized.")
        cells_fw.append(cell_fw)
        cells_bw.append(cell_bw)
    multi_cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    multi_cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
    with tf.variable_scope('BDGRU_rnn') as scope:
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_cells_fw, cell_bw=multi_cells_bw, inputs=x, sequence_length=seq_length, dtype=dtype,
            scope=scope)
    return outputs,states