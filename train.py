# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     train
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


def train(argv):
    batch_size = argv.batch_size
    init_learning_rate = argv.learning_rate
    decay_rate = argv.decay_rate
    class_num = argv.class_num
    keep_prob = argv.keep_prob
    display_steps = 100
    epoch_num = argv.epoch
    LOG_DIR = argv.output_log_dir
    top_first_accuracy = 0.0

    FEATURE_LEN = argv.base_num
    SIGNAL_LEN = argv.signal_num
    base_bytes = FEATURE_LEN * 1
    means_bytes = FEATURE_LEN * 4
    stds_bytes = FEATURE_LEN * 4
    sanum_bytes = FEATURE_LEN * 2
    signal_bytes = SIGNAL_LEN * 4
    label_bytes = 1
    rnames_bytes = argv.max_rname_len * 1
    record_len = base_bytes + means_bytes + stds_bytes + sanum_bytes + signal_bytes + label_bytes + rnames_bytes

    files = []
    for file in os.listdir(argv.input_dir):
        files.append(argv.input_dir + '/' + file)

    valid_files = []
    for file in os.listdir(argv.valid_dir):
        valid_files.append(argv.valid_dir + '/' + file)

    ## valid dataset
    valid_dataset = tf.data.FixedLengthRecordDataset(valid_files, record_len).map(
        lambda x: decode_line(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN,
                              rname_len=argv.max_rname_len))
    valid_dataset = valid_dataset.batch(batch_size).repeat()
    valid_iterator = valid_dataset.make_one_shot_iterator()
    valid_element = valid_iterator.get_next()

    model = Model(base_num=FEATURE_LEN, signal_num=SIGNAL_LEN, class_num=class_num)
    f = open('log.txt', 'w')
    iter_id = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch_id in range(epoch_num):
            start = time.time()
            if epoch_id == 0:
                learning_rate = init_learning_rate
            else:
                learning_rate = learning_rate * decay_rate

            ## train dataset
            dataset = tf.data.FixedLengthRecordDataset(files, record_len).map(
                lambda x: decode_line(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN,
                                      rname_len=argv.max_rname_len))
            dataset = dataset.shuffle(batch_size * 3).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            element = iterator.get_next()

            try:
                while True:
                    features, label = sess.run(element)
                    label = np.reshape(label, (label.shape[0]))
                    feed_dict = {model.base_int: features['base'],
                                 model.means: features['mean'],
                                 model.stds: features['std'],
                                 model.sanums: features['sanum'],
                                 model.signals: features['signal'],
                                 model.labels: label,
                                 model.lr: learning_rate,
                                 model.training: True,
                                 model.keep_prob: keep_prob}
                    train_loss, train_accuracy, _ = sess.run([model.loss, model.accuracy, model.train_opt],
                                                             feed_dict=feed_dict)
                    iter_id += 1
                    if iter_id % display_steps == 0:
                        test_accuracy_list = []
                        for j in range(3):
                            valid_features, valid_labels = sess.run(valid_element)
                            valid_labels = np.reshape(valid_labels, (valid_labels.shape[0]))
                            feed_dict = {model.base_int: valid_features['base'],
                                         model.means: valid_features['mean'],
                                         model.stds: valid_features['std'],
                                         model.sanums: valid_features['sanum'],
                                         model.signals: valid_features['signal'],
                                         model.labels: valid_labels,
                                         model.lr: learning_rate,
                                         model.training: False,
                                         model.keep_prob: 1.0}
                            test_loss, test_accuracy = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
                            test_accuracy_list.append(test_accuracy)
                        test_accuracy_list_mean = np.mean(test_accuracy_list)
                        if test_accuracy_list_mean > top_first_accuracy:
                            top_first_accuracy = test_accuracy_list_mean
                        end = time.time()
                        line = "Epoch: %d train_accuracy: %.3f test_accuracy: %.3f time_cost: %.2f" % (
                            epoch_id, train_accuracy, test_accuracy_list_mean, end - start)
                        print(line)
                        f.write(line + '\n')
                        start = time.time()
            except:
                print("epoch training end!")
            saver.save(sess, LOG_DIR + '/final.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-d', '--decay_rate', default=0.96, type=float)
    parser.add_argument('-c', '--class_num', default=2, type=int)
    parser.add_argument('-k', '--keep_prob', default=0.5, type=float)
    parser.add_argument('-e', '--epoch', default=15, type=int)
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-v', '--valid_dir', required=True)
    parser.add_argument('-o', '--output_log_dir', default='./log/')
    parser.add_argument('-x', '--base_num', default=17, type=int)
    parser.add_argument('-y', '--signal_num', default=120, type=int)
    parser.add_argument('-z', '--max_rname_len', default=100, type=int)

    argv = parser.parse_args()
    if os.path.exists(argv.output_log_dir):
        print('deleting the previous log direction...')
        shutil.rmtree(argv.output_log_dir)
        print('done')
    os.mkdir(argv.output_log_dir)

    train(argv)
