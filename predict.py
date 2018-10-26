# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     eval
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
import time
import os
import shutil
import numpy as np
from model import Model
from linedecode import decode_line
from sklearn import metrics


def predict(argv):
    batch_size = argv.batch_size
    init_learning_rate = argv.learning_rate
    class_num = argv.class_num
    # l1_scale = argv.l1_scale
    # l2_scale = argv.l2_scale
    MODLE_DIR = argv.model_dir
    MODEL_NAME = argv.model_name+'.ckpt'

    FEATURE_LEN = argv.base_num
    SIGNAL_LEN = argv.signal_num
    base_bytes = FEATURE_LEN * 1
    means_bytes = FEATURE_LEN * 4
    stds_bytes = FEATURE_LEN * 4
    sanum_bytes = FEATURE_LEN * 2
    signal_bytes = SIGNAL_LEN * 4
    label_bytes = 1
    rnames_bytes = argv.max_rname_len * 1
    record_len = base_bytes + means_bytes + stds_bytes + \
        sanum_bytes + signal_bytes + label_bytes + rnames_bytes

    files = []
    for file in os.listdir(argv.input_dir):
        files.append(argv.input_dir + '/' + file)

    model = Model(base_num=FEATURE_LEN,
                  signal_num=SIGNAL_LEN, class_num=class_num)

    fwrite = open(argv.result_file, 'w')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODLE_DIR + MODEL_NAME)
        accuracy_list = []
        dataset = tf.data.FixedLengthRecordDataset(files, record_len).map(
            lambda x: decode_line(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN, rname_len=argv.max_rname_len))
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()
        try:
            while True:
                features, label = sess.run(element)
                label = np.reshape(label, (label.shape[0]))
                # if label.shape[0] != batch_size:
                #     print('batch size:', label.shape[0])
                rnames = features['rname']
                feed_dict = {model.base_int: features['base'],
                             model.means: features['mean'],
                             model.stds: features['std'],
                             model.sanums: features['sanum'],
                             model.signals: features['signal'],
                             model.labels: label,
                             model.lr: init_learning_rate,
                             model.training: False,
                             model.keep_prob: 1.0}
                activation_logits, prediction = sess.run(
                    [model.activation_logits, model.prediction], feed_dict=feed_dict)
                accuracy = metrics.accuracy_score(
                    y_true=label, y_pred=prediction)
                for idx in range(label.shape[0]):
                    fwrite.write(bytes.decode(rnames[idx]) + '\t' + str(activation_logits[idx][0]) + '\t' + str(
                        activation_logits[idx][1]) + '\t' + str(prediction[idx]) + '\t' + str(label[idx]) + '\n')
                accuracy_list.append(accuracy)
        except:
            print("eval end!")
        print('total accuracy:', np.mean(accuracy_list))
        fwrite.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-c', '--class_num', default=2, type=int)
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--model_dir', required=True)
    parser.add_argument('-r', '--result_file', required=True)
    parser.add_argument('-n', '--model_name', default='5')
    parser.add_argument('-x', '--base_num', default=17, type=int)
    parser.add_argument('-y', '--signal_num', default=120, type=int)
    parser.add_argument('-z', '--max_rname_len', default=100, type=int)

    argv = parser.parse_args()
    predict(argv)
