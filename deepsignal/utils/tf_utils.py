from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from deepsignal.utils.process_utils import base2code_dna


def parse_a_line_b(value, base_num, signal_num):
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

    return bases, means, stds, sanum, signals, labels


def parse_a_line(line):
    def _kmer2code(kmer_bytes):
        return np.array([base2code_dna[x] for x in kmer_bytes.decode("utf-8")], np.int32)

    words = tf.decode_csv(line, [[""]] * 12, "\t")

    kmer = tf.py_func(_kmer2code, [words[6]], tf.int32)
    base_mean = tf.string_to_number(tf.string_split([words[7]], ",").values, tf.float32)
    base_std = tf.string_to_number(tf.string_split([words[8]], ",").values, tf.float32)
    base_signal_len = tf.string_to_number(tf.string_split([words[9]], ",").values, tf.int32)
    cent_signals = tf.string_to_number(tf.string_split([words[10]], ",").values, tf.float32)
    label = tf.string_to_number(words[11], tf.int32)

    return kmer, base_mean, base_std, base_signal_len, cent_signals, label
