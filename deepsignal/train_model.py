"""
train a deepsignal model. need two independent datasets for training and validating.
For CpG, Up to 20m (~10m positive and ~10m negative) samples for training and
10k (~5k positive and ~5k negative) samples for validating are sufficient.
"""

from __future__ import absolute_import

import tensorflow as tf
import argparse
import time
import os
import sys
import numpy as np
from sklearn import metrics
import re

from .model import Model
from .utils.process_utils import str2bool
from .utils.tf_utils import parse_a_line
from .utils.tf_utils import parse_a_line_b


def train(train_file, valid_file, model_dir, log_dir, kmer_len, cent_signals_len,
          batch_size, learning_rate, decay_rate, class_num, keep_prob, max_epoch_num,
          min_epoch_num, display_step, pos_weight, is_binary=False,
          is_rnn=True, is_base=True, is_cnn=True):
    train_start = time.time()

    train_file = os.path.abspath(train_file)
    valid_file = os.path.abspath(valid_file)

    model_regex = re.compile(r"bn_" + str(kmer_len) + "\.sn_" + str(cent_signals_len) +
                             "\.epoch_\d+\.ckpt*")
    model_prefix = "bn_" + str(kmer_len) + ".sn_" + str(cent_signals_len) + ".epoch_"
    model_suffix = '.ckpt'
    if os.path.exists(model_dir):
        # shutil.rmtree(model_dir)
        count = 0
        for mfile in os.listdir(model_dir):
            if model_regex.match(mfile) or mfile == "checkpoint":
                os.remove(model_dir + "/" + mfile)
                count += 1
        if count >= 1:
            print('the previous model ({} files) in model_directory deleted...'.format(count))
    else:
        os.mkdir(model_dir)

    train_log_txt = 'train.txt'
    valid_log_txt = 'valid.txt'
    if log_dir is not None:
        if os.path.exists(log_dir):
            # shutil.rmtree(log_dir)
            count = 0
            if os.path.exists(log_dir + '/' + train_log_txt):
                os.remove(log_dir + '/' + train_log_txt)
                count += 1
            if os.path.exists(log_dir + '/' + valid_log_txt):
                os.remove(log_dir + '/' + valid_log_txt)
                count += 1
            if count >= 1:
                print('the previous log file ({} files) in log_directory deleted...'.format(count))
        else:
            os.mkdir(log_dir)

    # train dataset
    if is_binary:
        FEATURE_LEN = kmer_len
        SIGNAL_LEN = cent_signals_len
        base_bytes = FEATURE_LEN * 1
        means_bytes = FEATURE_LEN * 4
        stds_bytes = FEATURE_LEN * 4
        sanum_bytes = FEATURE_LEN * 2
        signal_bytes = SIGNAL_LEN * 4
        label_bytes = 1
        record_len = base_bytes + means_bytes + stds_bytes + \
            sanum_bytes + signal_bytes + label_bytes
        dataset = tf.data.FixedLengthRecordDataset(train_file, record_len).map(
            lambda x: parse_a_line_b(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN))
    else:
        dataset = tf.data.TextLineDataset([train_file]).map(parse_a_line)
    dataset = dataset.shuffle(batch_size * 3).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    element = iterator.get_next()

    # valid dataset
    if is_binary:
        FEATURE_LEN = kmer_len
        SIGNAL_LEN = cent_signals_len
        base_bytes = FEATURE_LEN * 1
        means_bytes = FEATURE_LEN * 4
        stds_bytes = FEATURE_LEN * 4
        sanum_bytes = FEATURE_LEN * 2
        signal_bytes = SIGNAL_LEN * 4
        label_bytes = 1
        record_len = base_bytes + means_bytes + stds_bytes + \
                     sanum_bytes + signal_bytes + label_bytes
        valid_dataset = tf.data.FixedLengthRecordDataset(valid_file, record_len).map(
            lambda x: parse_a_line_b(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN))
    else:
        valid_dataset = tf.data.TextLineDataset([valid_file]).map(parse_a_line)
    valid_dataset = valid_dataset.batch(batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()
    valid_element = valid_iterator.get_next()

    model = Model(base_num=kmer_len,
                  signal_num=cent_signals_len, class_num=class_num, pos_weight=pos_weight,
                  is_cnn=is_cnn, is_base=is_base, is_rnn=is_rnn)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # sess.run(model.running_validation_vars_init)
        saver = tf.train.Saver()

        test_accu_best = 0.0
        for epoch_id in range(max_epoch_num):
            start = time.time()
            if epoch_id == 0 or epoch_id == 1:
                epoch_learning_rate = learning_rate
            else:
                epoch_learning_rate = learning_rate * decay_rate

            train_accuracy_total = []
            train_recall_total = []
            train_precision_total = []
            train_loss_total = []

            # validation
            test_accu_best_ep = 0.0
            test_accuracy_total = []
            test_recall_total = []
            test_precision_total = []
            test_loss_total = []

            iter_id = 0

            sess.run(iterator.initializer)

            while True:
                try:
                    b_kmer, b_base_mean, b_base_std, b_base_signal_len, \
                        b_cent_signals, b_label = sess.run(element)
                except tf.errors.OutOfRangeError:
                    break
                b_label = np.reshape(b_label, (b_label.shape[0]))
                feed_dict = {model.base_int: b_kmer,
                             model.means: b_base_mean,
                             model.stds: b_base_std,
                             model.sanums: b_base_signal_len,
                             model.signals: b_cent_signals,
                             model.labels: b_label,
                             model.lr: epoch_learning_rate,
                             model.training: True,
                             model.keep_prob: keep_prob}
                train_loss, _, train_prediction = sess.run(
                    [model.cost_pw, model.train_opt_pw, model.prediction], feed_dict=feed_dict)

                accu_batch = metrics.accuracy_score(
                    y_true=b_label, y_pred=train_prediction)
                if class_num == 2:
                    recall_batch = metrics.recall_score(
                        y_true=b_label, y_pred=train_prediction)
                    precision_batch = metrics.precision_score(
                        y_true=b_label, y_pred=train_prediction)
                else:
                    recall_batch = metrics.recall_score(
                        y_true=b_label, y_pred=train_prediction, average='micro')
                    precision_batch = metrics.precision_score(
                        y_true=b_label, y_pred=train_prediction, average='micro')

                train_loss_total.append(train_loss)
                train_accuracy_total.append(accu_batch)
                train_recall_total.append(recall_batch)
                train_precision_total.append(precision_batch)

                iter_id += 1
                if iter_id % display_step == 0:
                    # save the metrics of train
                    if log_dir is not None:
                        train_log = open(log_dir + '/' + train_log_txt, 'a')
                        t_log = "epoch:%d, iterid:%d, loss:%.3f, accuracy:%.3f, recall:%.3f, precision:%.3f\n" % \
                            (epoch_id, iter_id, np.mean(train_loss_total), np.mean(train_accuracy_total),
                             np.mean(train_recall_total), np.mean(train_precision_total))
                        train_log.write(t_log)
                        train_log.close()

                    sess.run(valid_iterator.initializer)
                    while True:
                        try:
                            v_kmer, v_base_mean, v_base_std, v_base_signal_len, \
                                v_cent_signals, v_label = sess.run(valid_element)
                        except tf.errors.OutOfRangeError:
                            break
                        v_label = np.reshape(v_label, (v_label.shape[0]))
                        feed_dict = {model.base_int: v_kmer,
                                     model.means: v_base_mean,
                                     model.stds: v_base_std,
                                     model.sanums: v_base_signal_len,
                                     model.signals: v_cent_signals,
                                     model.labels: v_label,
                                     model.lr: learning_rate,
                                     model.training: False,
                                     model.keep_prob: 1.0}
                        test_loss, test_prediction = sess.run(
                            [model.cost_pw, model.prediction], feed_dict=feed_dict)

                        accu_batch = metrics.accuracy_score(
                            y_true=v_label, y_pred=test_prediction)
                        if class_num == 2:
                            recall_batch = metrics.recall_score(
                                y_true=v_label, y_pred=test_prediction)
                            precision_batch = metrics.precision_score(
                                y_true=v_label, y_pred=test_prediction)
                        else:
                            recall_batch = metrics.recall_score(
                                y_true=v_label, y_pred=test_prediction, average='micro')
                            precision_batch = metrics.precision_score(
                                y_true=v_label, y_pred=test_prediction, average='micro')

                        test_loss_total.append(test_loss)
                        test_accuracy_total.append(accu_batch)
                        test_recall_total.append(recall_batch)
                        test_precision_total.append(precision_batch)

                    # save the metrics of test
                    if log_dir is not None:
                        valid_log = open(log_dir + '/' + valid_log_txt, 'a')
                        t_log = "epoch:%d, iterid:%d, loss:%.3f, accuracy:%.3f, recall:%.3f, precision:%.3f\n" % \
                            (epoch_id, iter_id, np.mean(test_loss_total), np.mean(test_accuracy_total),
                             np.mean(test_recall_total), np.mean(test_precision_total))
                        valid_log.write(t_log)
                        valid_log.close()

                    if np.mean(test_accuracy_total) > test_accu_best_ep:
                        test_accu_best_ep = np.mean(test_accuracy_total)
                        if test_accu_best_ep > test_accu_best:
                            saver.save(sess, "/".join([model_dir,
                                                       model_prefix + str(epoch_id) + model_suffix]))

                    end = time.time()
                    line = "epoch: %d, iterid: %d\n train_loss: %.3f, valid_loss: %.3f, train_accuracy: %.3f, " \
                           "valid_accuracy: %.3f, curr_epoch_best_accuracy: %.3f, " \
                           "time_cost: %.2fs" % (epoch_id, iter_id, np.mean(train_loss_total),
                                                 np.mean(test_loss_total),
                                                 np.mean(train_accuracy_total),
                                                 np.mean(test_accuracy_total),
                                                 test_accu_best_ep, end - start)
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()

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

            if test_accu_best_ep > test_accu_best:
                test_accu_best = test_accu_best_ep
                sys.stdout.write("================ epoch %d best accuracy: %.3f, "
                                 "best accuracy: %.3f\n" % (epoch_id,
                                                            test_accu_best_ep,
                                                            test_accu_best))
                sys.stdout.flush()
            else:
                sys.stdout.write("================ epoch %d best accuracy: %.3f, "
                                 "best accuracy: %.3f\n" % (epoch_id,
                                                            test_accu_best_ep,
                                                            test_accu_best))
                sys.stdout.flush()
                if epoch_id >= min_epoch_num - 1:
                    break
    sys.stdout.write("training finished, costs %.1f seconds..\n" % (time.time() - train_start))


def main():
    parser = argparse.ArgumentParser("train a model, need two independent datasets for training and validating")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--train_file", action="store", type=str, required=True,
                         help="file contains samples for training, from extract_features.py. "
                              "The file should contain shuffled positive and negative samples. "
                              "For CpG, Up to 20m (~10m positive and ~10m negative) samples are sufficient.")
    p_input.add_argument("--valid_file", action="store", type=str, required=True,
                         help="file contains samples for testing, from extract_features.py. "
                              "The file should contain shuffled positive and negative samples. "
                              "For CpG, 10k (~5k positive and ~5k negative) samples are sufficient.")
    p_input.add_argument("--is_binary", action="store", type=str, required=False,
                         default="no", choices=["yes", "no"],
                         help="are the train_file and valid_file in binary format or not? "
                              "'yes' or 'no', default no. "
                              "(for binary format, see scripts/generate_binary_feature_file.py)")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--model_dir", "-o", action="store", type=str, required=True,
                          help="directory for saving the trained model")
    p_output.add_argument("--log_dir", "-g", action="store", type=str, required=False,
                          default=None,
                          help="directory for saving the training log")

    p_train = parser.add_argument_group("TRAIN")
    p_train.add_argument('--is_cnn', type=str, default='yes', required=False,
                         help="using inception module of deepsignal or not")
    p_train.add_argument('--is_base', type=str, default='yes', required=False,
                         help="using base features in BiLSTM module or not")
    p_train.add_argument('--is_rnn', type=str, default='yes', required=False,
                         help="using BiLSTM module of deepsignal or not")

    p_train.add_argument("--kmer_len", "-x", action="store", default=17, type=int, required=False,
                         help="base num of the kmer, default 17")
    p_train.add_argument("--cent_signals_len", "-y", action="store", default=360, type=int, required=False,
                         help="the number of central signals of the kmer to be used, default 360")

    p_train.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                         action="store", help="batch size, default 512")
    p_train.add_argument("--learning_rate", "-l", default=0.001, type=float, required=False,
                         action="store", help="init learning rate, default 0.001")
    p_train.add_argument("--decay_rate", "-d", default=0.1, type=float, required=False,
                         action="store", help="decay rate, default 0.1")
    p_train.add_argument("--class_num", "-c", action="store", default=2, type=int, required=False,
                         help="class num, default 2")
    p_train.add_argument("--keep_prob", action="store", default=0.5, type=float,
                         required=False, help="keep prob, default 0.5")
    p_train.add_argument("--max_epoch_num", action="store", default=10, type=int,
                         required=False, help="max epoch num, default 10")
    p_train.add_argument("--min_epoch_num", action="store", default=5, type=int,
                         required=False, help="min epoch num, default 5")
    p_train.add_argument("--display_step", action="store", default=100, type=int,
                         required=False, help="display step, default 100")
    p_train.add_argument("--pos_weight", action="store", default=1.0, type=float,
                         required=False, help="pos_weight in loss function: "
                                              "tf.nn.weighted_cross_entropy_with_logits, "
                                              "for imbalanced training samples, default 1. "
                                              "If |pos samples| : |neg samples| = 1:3, set pos_weight to 3.")

    args = parser.parse_args()

    train_file = args.train_file
    valid_file = args.valid_file
    is_binary = str2bool(args.is_binary)

    model_dir = os.path.abspath(args.model_dir)
    if args.log_dir is not None:
        log_dir = os.path.abspath(args.log_dir)
    else:
        log_dir = None

    is_cnn = str2bool(args.is_cnn)
    is_base = str2bool(args.is_base)
    is_rnn = str2bool(args.is_rnn)

    kmer_len = args.kmer_len
    cent_signals_len = args.cent_signals_len
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    class_num = args.class_num
    keep_prob = args.keep_prob
    max_epoch_num = args.max_epoch_num
    min_epoch_num = args.min_epoch_num
    display_step = args.display_step
    pos_weight = args.pos_weight

    train(train_file, valid_file, model_dir, log_dir, kmer_len, cent_signals_len,
          batch_size, learning_rate, decay_rate, class_num, keep_prob, max_epoch_num,
          min_epoch_num, display_step, pos_weight, is_binary, is_rnn, is_base, is_cnn)


if __name__ == '__main__':
    sys.exit(main())
