"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
"""

import tensorflow as tf
import argparse
import os
import numpy as np
from model import Model
from sklearn import metrics

import multiprocessing as mp
import time

from utils.process_utils import base2code_dna
from utils.process_utils import code2base_dna
from utils.process_utils import str2bool

queen_size_border = 1000
time_wait = 5


def _read_features_file(features_file, features_batch_q, batch_num=512):
    with open(features_file, "r") as rf:
        # chromosome, pos, strand, pos_in_strand, read_name, read_strand = [], [], [], [], [], []
        sampleinfo = []  # contains: chromosome, pos, strand, pos_in_strand, read_name, read_strand
        kmers = []
        base_means = []
        base_stds = []
        base_signal_lens = []
        cent_signals = []
        labels = []

        for line in rf:
            words = line.strip().split("\t")

            sampleinfo.append("\t".join(words[0:6]))

            kmers.append([base2code_dna[x] for x in words[6]])
            base_means.append([float(x) for x in words[7].split(",")])
            base_stds.append([float(x) for x in words[8].split(",")])
            base_signal_lens.append([int(x) for x in words[9].split(",")])
            cent_signals.append([float(x) for x in words[10].split(",")])
            labels.append(int(words[11]))

            if len(sampleinfo) == batch_num:
                features_batch_q.put((sampleinfo, kmers, base_means, base_stds,
                                      base_signal_lens, cent_signals, labels))
                if features_batch_q.qsize() >= queen_size_border:
                    time.sleep(time_wait)
                sampleinfo = []
                kmers = []
                base_means = []
                base_stds = []
                base_signal_lens = []
                cent_signals = []
                labels = []
        if len(sampleinfo) > 0:
            features_batch_q.put((sampleinfo, kmers, base_means, base_stds,
                                  base_signal_lens, cent_signals, labels))
    features_batch_q.put("kill")


def _call_mods(init_learning_rate, class_num, model_path,
               base_num, signal_num, features_batch_q, pred_str_q,
               success_file):

    model = Model(base_num=base_num,
                  signal_num=signal_num, class_num=class_num)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        sess.run(tf.local_variables_initializer())

        accuracy_list = []
        while True:
            if os.path.exists(success_file):
                break

            if features_batch_q.empty():
                time.sleep(time_wait)
                continue

            features_batch = features_batch_q.get()
            if features_batch == "kill":
                open(success_file, 'w').close()
                break

            sampleinfo, kmers, base_means, base_stds, base_signal_lens, \
                cent_signals, labels = features_batch
            labels = np.reshape(labels, (len(labels)))

            feed_dict = {model.base_int: kmers,
                         model.means: base_means,
                         model.stds: base_stds,
                         model.sanums: base_signal_lens,
                         model.signals: cent_signals,
                         model.labels: labels,
                         model.lr: init_learning_rate,
                         model.training: False,
                         model.keep_prob: 1.0}
            activation_logits, prediction = sess.run(
                [model.activation_logits, model.prediction], feed_dict=feed_dict)
            accuracy = metrics.accuracy_score(
                y_true=labels, y_pred=prediction)

            features_str = []
            for idx in range(labels.shape[0]):
                # chromosome, pos, strand, pos_in_strand, read_name, read_strand, prob_0, prob_1, called_label, seq
                features_str.append("\t".join([sampleinfo[idx], str(activation_logits[idx][0]),
                                               str(activation_logits[idx][1]), str(prediction[idx]),
                                               ''.join([code2base_dna[x] for x in kmers[idx]])]))
            pred_str_q.put(features_str)

            accuracy_list.append(accuracy)
        # print("eval end!")
        print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))


def _write_predstr_to_file(write_fp, predstr_q):
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep()
            if predstr_q.empty():
                time.sleep(time_wait)
            pred_str = predstr_q.get()
            if pred_str == "kill":
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def call_mods(input_path, is_recursive, model_path, result_file, kmer_len, cent_signals_len,
              batch_size, learning_rate, class_num, nproc):
    start = time.time()

    input_path = os.path.abspath(input_path)
    success_file = input_path.rstrip("/") + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)

    if os.path.isdir(input_path):
        # fast5_files
        pass
    else:
        features_batch_q = mp.Queue()
        p_rf = mp.Process(target=_read_features_file, args=(input_path, features_batch_q, batch_size))
        p_rf.daemon = True
        p_rf.start()

        pred_str_q = mp.Queue()

        predstr_procs = []
        if nproc > 2:
            nproc -= 2
        for _ in range(nproc):
            p = mp.Process(target=_call_mods, args=(learning_rate, class_num, model_path,
                                                    kmer_len, cent_signals_len, features_batch_q,
                                                    pred_str_q, success_file))
            p.daemon = True
            p.start()
            predstr_procs.append(p)

        print("write_process started..")
        p_w = mp.Process(target=_write_predstr_to_file, args=(result_file, pred_str_q))
        p_w.daemon = True
        p_w.start()

        for p in predstr_procs:
            p.join()
        pred_str_q.put("kill")

        p_rf.join()

        p_w.join()

    if os.path.exists(success_file):
        os.remove(success_file)
    print("call_mods costs %.1f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", action="store", type=str,
                        required=True,
                        help="the input path, can be a directory of fast5 files or "
                             "a signal_feature file from extract_features.py")
    parser.add_argument("--recursively", "-r", action="store", type=str, required=False,
                        default='yes', help='is to find fast5 files from fast5_dir recursively. '
                                            'default true, t, yes, 1')
    parser.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")
    parser.add_argument("--learning_rate", "-l", default=0.001, type=float, required=False,
                        action="store", help="learning rate, default 0.001")
    parser.add_argument("--class_num", "-c", action="store", default=2, type=int, required=False,
                        help="class num, default 2")
    parser.add_argument("--model_path", "-m", action="store", type=str, required=True,
                        help="the trained model file path (.ckpt)")
    parser.add_argument("--result_file", "-o", action="store", type=str, required=True,
                        help="the file path to save the predicted result")
    parser.add_argument("--kmer_len", "-x", action="store", default=17, type=int, required=False,
                        help="base num of the kmer, default 17")
    parser.add_argument("--cent_signals_len", "-y", action="store", default=360, type=int, required=False,
                        help="the number of central signals of the kmer to be used, default 360")
    parser.add_argument("--nproc", "-p", action="store", type=int, default=1,
                        required=False, help="number of processes to be used, default 1.")

    args = parser.parse_args()

    input_path = args.input_path
    is_recursive = str2bool(args.recursively)
    model_path = args.model_path
    result_file = args.result_file

    kmer_len = args.kmer_len
    cent_signals_len = args.cent_signals_len

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    class_num = args.class_num

    nproc = args.nproc

    call_mods(input_path, is_recursive, model_path, result_file, kmer_len, cent_signals_len,
              batch_size, learning_rate, class_num, nproc)


if __name__ == '__main__':
    main()
