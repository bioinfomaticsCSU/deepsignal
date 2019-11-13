from __future__ import absolute_import

import argparse
import tensorflow as tf
import time
import os
import sys
import numpy as np
from sklearn import metrics

from .model import Model
from .utils.process_utils import str2bool
from .utils.process_utils import random_select_file_rows_s
from .utils.process_utils import random_select_file_rows
from .utils.process_utils import count_line_num
from .utils.process_utils import concat_two_files
from .utils.process_utils import extract
from .utils.tf_utils import parse_a_line_b


def _convert_txt2bin(train_file, args):
    fname, fext = os.path.splitext(train_file)
    train_file_bin = fname + ".bin"
    # txt2bin ===
    format_string = '<' + str(args.seq_len) + 'B' + str(args.seq_len) + 'f' + str(args.seq_len) + 'f' + str(
        args.seq_len) + 'H' + str(args.cent_signals_len) + 'f' + '1B'
    extract(train_file, train_file_bin, format_string)
    return train_file_bin


def train_1time(train_file_bin, valid_file_bin, valid_lidxs, modeltime, args):
    # ===========
    # dataset = tf.data.TextLineDataset([train_file]).map(_parse_a_line)
    FEATURE_LEN = args.seq_len
    SIGNAL_LEN = args.cent_signals_len
    base_bytes = FEATURE_LEN * 1
    means_bytes = FEATURE_LEN * 4
    stds_bytes = FEATURE_LEN * 4
    sanum_bytes = FEATURE_LEN * 2
    signal_bytes = SIGNAL_LEN * 4
    label_bytes = 1
    record_len = base_bytes + means_bytes + stds_bytes + \
        sanum_bytes + signal_bytes + label_bytes
    dataset = tf.data.FixedLengthRecordDataset(train_file_bin, record_len).map(
        lambda x: parse_a_line_b(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN))

    # dataset = dataset.batch(args.batch_size)
    dataset = dataset.shuffle(args.batch_size * 3).batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    element = iterator.get_next()

    model = Model(base_num=args.seq_len,
                  signal_num=args.cent_signals_len, class_num=args.class_num,
                  is_cnn=str2bool(args.is_cnn), is_base=str2bool(args.is_base),
                  is_rnn=str2bool(args.is_rnn),
                  pos_weight=args.pos_weight,
                  model_prefix=args.model_prefix + str(modeltime))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # saver = tf.train.Saver()

        for epoch_id in range(args.epoch_num):
            start = time.time()
            if epoch_id == 0 or epoch_id == 1:
                epoch_learning_rate = args.lr
            else:
                epoch_learning_rate = args.lr * args.decay_rate

            sess.run(iterator.initializer)
            iter_id = 0

            test_accus = []
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
                             model.keep_prob: args.keep_prob}
                train_loss, _, train_prediction = sess.run(
                    [model.cost_pw, model.train_opt_pw, model.prediction], feed_dict=feed_dict)

                iter_id += 1
                if iter_id % args.step_interval == 0:
                    accu_batch = metrics.accuracy_score(
                        y_true=b_label, y_pred=train_prediction)
                    recall_batch = metrics.recall_score(
                        y_true=b_label, y_pred=train_prediction)
                    precision_batch = metrics.precision_score(
                        y_true=b_label, y_pred=train_prediction)

                    test_accus.append(accu_batch)

                    endtime = time.time()

                    print('Epoch [{}/{}], Step {}, Loss: {:.4f}, '
                          'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                          'Time: {:.2f}s'
                          .format(epoch_id + 1, args.epoch_num, iter_id, train_loss,
                                  accu_batch, precision_batch, recall_batch, endtime - start))
                    sys.stdout.flush()
                    start = time.time()

            if np.mean(test_accus) >= 0.95:
                break

        vdataset = tf.data.FixedLengthRecordDataset(valid_file_bin, record_len).map(
            lambda x: parse_a_line_b(value=x, base_num=FEATURE_LEN, signal_num=SIGNAL_LEN))

        vdataset = vdataset.batch(args.batch_size)
        viterator = vdataset.make_initializable_iterator()
        velement = viterator.get_next()

        sess.run(viterator.initializer)
        vaccus = []
        vprecs = []
        vrecas = []
        start = time.time()
        iter_id = 0
        lineidx_cnt = 0
        idx2aclogits = {}
        while True:
            try:
                v_kmer, v_base_mean, v_base_std, v_base_signal_len, \
                    v_cent_signals, v_label = sess.run(velement)
            except tf.errors.OutOfRangeError:
                break
            v_label = np.reshape(v_label, (v_label.shape[0]))
            # lineidx = np.reshape(lineidx, (lineidx.shape[0]))
            feed_dict = {model.base_int: v_kmer,
                         model.means: v_base_mean,
                         model.stds: v_base_std,
                         model.sanums: v_base_signal_len,
                         model.signals: v_cent_signals,
                         model.labels: v_label,
                         model.lr: args.lr,
                         model.training: False,
                         model.keep_prob: 1.0}
            activation_logits, prediction = sess.run(
                [model.activation_logits1, model.prediction], feed_dict=feed_dict)

            vaccu_batch = metrics.accuracy_score(
                y_true=v_label, y_pred=prediction)
            vreca_batch = metrics.recall_score(
                y_true=v_label, y_pred=prediction)
            vprec_batch = metrics.precision_score(
                y_true=v_label, y_pred=prediction)
            vaccus.append(vaccu_batch)
            vprecs.append(vprec_batch)
            vrecas.append(vreca_batch)

            iter_id += 1
            if iter_id % args.step_interval == 0:
                endtime = time.time()
                print('===test Step {}, '
                      'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                      'Time: {:.2f}s'
                      .format(iter_id, vaccu_batch, vprec_batch, vreca_batch, endtime - start))
                sys.stdout.flush()

            for alogit in activation_logits:
                idx2aclogits[valid_lidxs[lineidx_cnt]] = alogit
                lineidx_cnt += 1

        endtime = time.time()
        print("===test total Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
              "Time: {:.2f}s".format(np.mean(vaccus), np.mean(vprecs), np.mean(vrecas),
                                     endtime - start))
        return idx2aclogits


def train_rounds(train_file, iterstr, args):
    print("\n##########Train Cross Rank##########")
    total_num = count_line_num(train_file, False)
    half_num = total_num // 2
    fname, fext = os.path.splitext(train_file)
    idxs2logtis_all = {}
    for i in range(0, total_num):
        idxs2logtis_all[i] = []

    for i in range(0, args.rounds):
        print("\n##########Train Cross Rank, Iter {}, Round {}##########".format(iterstr, i+1))
        train_file1 = fname + ".half1" + fext
        train_file2 = fname + ".half2" + fext
        lidxs1, lidxs2 = random_select_file_rows_s(train_file, train_file1, train_file2,
                                                   half_num, False)
        train_file1_bin = _convert_txt2bin(train_file1, args)
        train_file2_bin = _convert_txt2bin(train_file2, args)

        idxs22logits = train_1time(train_file1_bin, train_file2_bin, lidxs2, iterstr + str(i) + str(1), args)
        idxs12logits = train_1time(train_file2_bin, train_file1_bin, lidxs1, iterstr + str(i) + str(2), args)
        for idx in idxs22logits.keys():
            idxs2logtis_all[idx].append(idxs22logits[idx])
        for idx in idxs12logits.keys():
            idxs2logtis_all[idx].append(idxs12logits[idx])

        os.remove(train_file1)
        os.remove(train_file2)
        os.remove(train_file1_bin)
        os.remove(train_file2_bin)
    print("\n##########Train Cross Rank End!##########")
    return idxs2logtis_all


def clean_samples(train_file, idx2logits, score_cf=0.5):
    # clean train samples ===
    print("###### clean the samples ######")
    idx2probs = dict()
    for idx in idx2logits.keys():
        probs = idx2logits[idx]
        meanprob = float(sum(probs)) / len(probs)
        stdprob = np.std(probs)
        idx2probs[idx] = [meanprob, stdprob]

    idx2prob_pos, idx2prob_neg = [], []
    with open(train_file, 'r') as rf:
        linecnt = 0
        for line in rf:
            words = line.strip().split("\t")
            label = int(words[-1])
            if label == 1:
                idx2prob_pos.append((linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1]))
            else:
                idx2prob_neg.append((linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1]))
            linecnt += 1

    print("there are {} positive, {} negative samples in total;".format(len(idx2prob_pos),
                                                                        len(idx2prob_neg)))

    # idx2prob_neg = sorted(idx2prob_neg, key=lambda x: x[1])
    idx2prob_pos = sorted(idx2prob_pos, key=lambda x: x[1], reverse=True)

    pos_hc, neg_hc = set(), set()
    # for idx2prob in idx2prob_neg:
    #     if idx2prob[1] < score_cf:
    #         neg_hc.add(idx2prob[0])
    for idx2prob in idx2prob_pos:
        if idx2prob[1] > score_cf:
            pos_hc.add(idx2prob[0])

    left_ratio = float(len(pos_hc)) / len(idx2prob_pos)
    print("{} ({}) high quality positive samples left, "
          "{} high quality negative samples left".format(len(pos_hc),
                                                         left_ratio,
                                                         len(neg_hc)))

    # re-write train set
    fname, fext = os.path.splitext(train_file)
    # train_clean_neg_file = fname + ".neg.cf" + str(score_cf) + fext
    train_clean_pos_file = fname + ".pos.cf" + str(score_cf) + fext

    # wfn = open(train_clean_neg_file, 'w')
    wfp = open(train_clean_pos_file, 'w')
    lidx = 0
    with open(train_file, 'r') as rf:
        for line in rf:
            # lidx = int(line.strip().split("\t")[12])
            if lidx in pos_hc:
                wfp.write(line)
            # elif lidx in neg_hc:
            #     wfn.write(line)
            lidx += 1
    # wfn.close()
    wfp.close()

    print("###### clean the samples end! ######")

    # return train_clean_pos_file, train_clean_neg_file
    return train_clean_pos_file, left_ratio


def _get_all_negative_samples(train_file):
    fname, fext = os.path.splitext(train_file)
    train_neg_file = fname + ".neg_all" + fext

    wf = open(train_neg_file, "w")
    with open(train_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            label = int(words[-1])
            if label == 0:
                wf.write(line)
    wf.close()
    return train_neg_file


def denoise(args):
    total_start = time.time()

    iterations = args.iterations

    train_file = args.train_file

    # filter neg samples ===
    train_neg_file = _get_all_negative_samples(train_file)

    for iter_c in range(iterations):
        print("###### cross rank to clean samples, iter: {} ######".format(iter_c + 1))
        # cross rank
        iterstr = str(iter_c + 1)
        idxs2logtis_all = train_rounds(train_file, iterstr, args)
        train_clean_pos_file, left_ratio = clean_samples(train_file, idxs2logtis_all, args.score_cf)

        # concat new train_file
        pos_num = count_line_num(train_clean_pos_file)
        fname, fext = os.path.splitext(train_neg_file)
        train_clean_neg_file = fname + ".r" + str(pos_num) + fext
        random_select_file_rows(train_neg_file, train_clean_neg_file, None, pos_num)

        fname, fext = os.path.splitext(args.train_file)
        train_file = fname + ".denoise" + str(iter_c + 1) + fext
        concat_two_files(train_clean_pos_file, train_clean_neg_file, concated_fp=train_file)

        os.remove(train_clean_neg_file)
        os.remove(train_clean_pos_file)

        if left_ratio > 0.99:
            break

    os.remove(train_neg_file)
    total_end = time.time()
    print("###### training totally costs {:.2f} seconds".format(total_end - total_start))


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")


def main():
    parser = argparse.ArgumentParser("train cross rank")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--model_prefix', type=str, default="model",
                        required=False)

    parser.add_argument('--is_cnn', type=str, default='no', required=False)
    parser.add_argument('--is_base', type=str, default='no', required=False)
    parser.add_argument('--is_rnn', type=str, default='yes', required=False)

    parser.add_argument('--seq_len', type=int, default=17, required=False)
    parser.add_argument('--cent_signals_len', type=int, default=360, required=False)
    parser.add_argument('--layer_num', type=int, default=3, required=False)
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--batch_size', type=int, default=512, required=False)

    parser.add_argument('--lr', type=float, default=0.001, required=False,
                        help="learning rate")

    parser.add_argument('--decay_rate', type=float, default=0.1, required=False)
    parser.add_argument('--keep_prob', action="store", default=0.5, type=float,
                        required=False, help="keep prob, default 0.5")

    parser.add_argument('--iterations', type=int, default=6, required=False)
    parser.add_argument('--epoch_num', type=int, default=5, required=False)
    parser.add_argument('--step_interval', type=int, default=100, required=False)
    parser.add_argument('--rounds', type=int, default=5, required=False)
    parser.add_argument("--score_cf", type=float, default=0.5,
                        required=False,
                        help="score cutoff")

    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    denoise(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime-total_start))


if __name__ == '__main__':
    main()
