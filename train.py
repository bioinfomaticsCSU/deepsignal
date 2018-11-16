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
from sklearn import metrics


def train(argv):
    batch_size = argv.batch_size
    init_learning_rate = argv.learning_rate
    decay_rate = argv.decay_rate
    class_num = argv.class_num
    keep_prob = argv.keep_prob
    display_steps = argv.display_steps
    epoch_num = argv.epoch
    MODEL_DIR = argv.output_model_dir
    LOG_DIR = argv.output_log_dir

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

    valid_files = []
    for file in os.listdir(argv.valid_dir):
        valid_files.append(argv.valid_dir + '/' + file)

    # valid dataset
    valid_dataset = tf.data.FixedLengthRecordDataset(valid_files, record_len).map(
        lambda x: decode_line(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN,
                              rname_len=argv.max_rname_len))
    valid_dataset = valid_dataset.batch(batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()
    valid_element = valid_iterator.get_next()

    # train dataset
    dataset = tf.data.FixedLengthRecordDataset(files, record_len).map(
        lambda x: decode_line(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN,
                              rname_len=argv.max_rname_len))
    dataset = dataset.shuffle(batch_size * 3).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    element = iterator.get_next()

    model = Model(base_num=FEATURE_LEN,
                  signal_num=SIGNAL_LEN, class_num=class_num)
    iter_id = 0
    if os.path.exists(LOG_DIR+'/'+'train.txt'):
        os.remove(LOG_DIR+'/'+'train.txt')
    if os.path.exists(LOG_DIR+'/'+'test.txt'):
        os.remove(LOG_DIR+'/'+'test.txt')
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # sess.run(model.running_validation_vars_init)
        saver = tf.train.Saver()

        for epoch_id in range(epoch_num):
            start = time.time()
            if epoch_id == 0 or epoch_id == 1:
                learning_rate = init_learning_rate
            else:
                learning_rate = init_learning_rate * decay_rate

            train_accuracy_total = []
            train_recall_total = []
            train_precision_total = []
            train_loss_total = []

            test_accuracy_total = []
            test_recall_total = []
            test_precision_total = []
            test_loss_total = []

            iter_id = 0

            sess.run(iterator.initializer)

            while True:
                try:
                    features, label = sess.run(element)
                except tf.errors.OutOfRangeError:
                    break
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
                train_loss, _, train_prediction = sess.run(
                    [model.loss, model.train_opt, model.prediction], feed_dict=feed_dict)
                # TODO: accuracy recall precision
                accu_batch = metrics.accuracy_score(
                    y_true=label, y_pred=train_prediction)
                recall_batch = metrics.recall_score(
                    y_true=label, y_pred=train_prediction)
                precision_batch = metrics.precision_score(
                    y_true=label, y_pred=train_prediction)

                train_loss_total.append(train_loss)
                train_accuracy_total.append(accu_batch)
                train_recall_total.append(recall_batch)
                train_precision_total.append(precision_batch)

                iter_id += 1
                if iter_id % display_steps == 0:
                    # save the metrics of train
                    train_log = open(LOG_DIR+'/'+'train.txt', 'a')
                    t_log = "epoch:%d, iterid:%d, loss:%.3f, accuracy:%.3f, recall:%.3f, precision:%.3f\n" % (epoch_id, iter_id, np.mean(train_loss_total), np.mean(
                        train_accuracy_total), np.mean(train_recall_total), np.mean(train_precision_total))
                    train_log.write(t_log)
                    train_log.close()

                    sess.run(valid_iterator.initializer)
                    while True:
                        try:
                            valid_features, valid_labels = sess.run(
                                valid_element)
                        except tf.errors.OutOfRangeError:
                            break
                        valid_labels = np.reshape(
                            valid_labels, (valid_labels.shape[0]))
                        feed_dict = {model.base_int: valid_features['base'],
                                     model.means: valid_features['mean'],
                                     model.stds: valid_features['std'],
                                     model.sanums: valid_features['sanum'],
                                     model.signals: valid_features['signal'],
                                     model.labels: valid_labels,
                                     model.lr: learning_rate,
                                     model.training: False,
                                     model.keep_prob: 1.0}
                        test_loss, test_prediction = sess.run(
                            [model.loss, model.prediction], feed_dict=feed_dict)
                        # TODO: accuracy recall precision
                        accu_batch = metrics.accuracy_score(
                            y_true=valid_labels, y_pred=test_prediction)
                        recall_batch = metrics.recall_score(
                            y_true=valid_labels, y_pred=test_prediction)
                        precision_batch = metrics.precision_score(
                            y_true=valid_labels, y_pred=test_prediction)

                        test_loss_total.append(test_loss)
                        test_accuracy_total.append(accu_batch)
                        test_recall_total.append(recall_batch)
                        test_precision_total.append(precision_batch)

                    # save the metrics of test
                    valid_log = open(LOG_DIR+'/'+'test.txt', 'a')
                    t_log = "epoch:%d, iterid:%d, loss:%.3f, accuracy:%.3f, recall:%.3f, precision:%.3f\n" % (epoch_id, iter_id, np.mean(test_loss_total), np.mean(
                        test_accuracy_total), np.mean(test_recall_total), np.mean(test_precision_total))
                    valid_log.write(t_log)
                    valid_log.close()

                    end = time.time()
                    line = "Epoch: %d, iterid: %d\n train_loss: %.3f test_loss: %.3f train_accuracy: %.3f test_accuracy: %.3f time_cost: %.2f" % (
                        epoch_id, iter_id, np.mean(train_loss_total), np.mean(test_loss_total), np.mean(train_accuracy_total), np.mean(test_accuracy_total), end - start)
                    print(line)
                    # reset train metrics
                    train_accuracy_total = []
                    train_recall_total = []
                    train_precision_total = []
                    train_loss_total = []

                    # reset valid metrics
                    test_accuracy_total = []
                    test_recall_total = []
                    test_precision_total = []
                    test_loss_total = []

                    start = time.time()
                    saver.save(sess, MODEL_DIR+'/'+str(epoch_id) + '.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-d', '--decay_rate', default=0.1, type=float)
    parser.add_argument('-c', '--class_num', default=2, type=int)
    parser.add_argument('-k', '--keep_prob', default=0.5, type=float)
    parser.add_argument('-e', '--epoch', default=15, type=int)
    parser.add_argument('-s', '--display_steps', default=100, type=int)
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-v', '--valid_dir', required=True)
    parser.add_argument('-o', '--output_model_dir', default='./logs/')
    parser.add_argument('-g', '--output_log_dir', default='./models/')
    parser.add_argument('-x', '--base_num', default=17, type=int)
    parser.add_argument('-y', '--signal_num', default=120, type=int)
    parser.add_argument('-z', '--max_rname_len', default=100, type=int)

    argv = parser.parse_args()
    if os.path.exists(argv.output_model_dir):
        print('deleting the previous model direction...')
        shutil.rmtree(argv.output_model_dir)
        print('done')
    os.mkdir(argv.output_model_dir)

    if os.path.exists(argv.output_log_dir):
        print('deleting the previous log direction...')
        shutil.rmtree(argv.output_log_dir)
        print('done')
    os.mkdir(argv.output_log_dir)

    train(argv)
