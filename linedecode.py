# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     linedecode
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
import argparse

DNA_DIC = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def decode_line(value, base_num, signal_num, rname_len):
    def ascii2str(ascii_lst):
        return [''.join([chr(v) for v in ascii_lst]).strip()]

    vec = tf.decode_raw(value, tf.int8)

    bases = tf.cast(tf.reshape(tf.strided_slice(vec, [0], [base_num]), [base_num]), dtype=tf.int32)
    means = tf.bitcast(
        tf.reshape(tf.strided_slice(vec, [base_num], [base_num + base_num * 4]), [base_num, 4]),
        type=tf.float32)
    stds = tf.bitcast(
        tf.reshape(tf.strided_slice(vec, [base_num * 5], [base_num * 5 + base_num * 4]), [base_num, 4]),
        type=tf.float32)
    sanum = tf.cast(tf.bitcast(
        tf.reshape(tf.strided_slice(vec, [base_num * 9], [base_num * 9 + base_num * 2]), [base_num, 2]),
        type=tf.int16), dtype=tf.int32)
    signals = tf.bitcast(
        tf.reshape(tf.strided_slice(vec, [base_num * 11], [base_num * 11 + 4 * signal_num]),
                   [signal_num, 4]), type=tf.float32)
    labels = tf.cast(
        tf.reshape(tf.strided_slice(vec, [base_num * 11 + signal_num * 4], [base_num * 11 + signal_num * 4 + 1]),
                   [1]),
        dtype=tf.int32)
    rnames = tf.py_func(ascii2str, [tf.cast(tf.reshape(tf.strided_slice(vec, [base_num * 11 + signal_num * 4 + 1],
                                                                        [
                                                                            base_num * 11 + signal_num * 4 + 1 + rname_len]),
                                                       [rname_len]), dtype=tf.int32)], tf.string)

    feature_names = ['base', 'mean', 'std', 'sanum', 'signal', 'rname']
    features = [bases, means, stds, sanum, signals, rnames]
    d = dict(zip(feature_names, features)), labels
    return d


if __name__ == "__main__":
    files = './training/training.bin'
    MAX_RNAME_LEN = 100
    base_num = 17
    signal_num = 120
    base_bytes = base_num * 1
    means_bytes = base_num * 4
    stds_bytes = base_num * 4
    sanum_bytes = base_num * 2
    signal_bytes = signal_num * 4
    label_bytes = 1
    rnames_bytes = MAX_RNAME_LEN
    record_len = base_bytes + means_bytes + stds_bytes + sanum_bytes + signal_bytes + label_bytes + rnames_bytes
    with tf.Session() as sess:
        dataset = tf.data.FixedLengthRecordDataset(files, record_len).map(
            lambda x: decode_line(value=x, base_num=base_num, signal_num=signal_num, rname_len=MAX_RNAME_LEN))
        dataset = dataset.batch(10)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()
        # while True:
        features, labels = sess.run(element)
        print(features['rname'])
        print(features['base'])
        print(features['mean'])
        print(features['std'])
        print(features['sanum'])
        print(features['signal'])
        print(labels)
