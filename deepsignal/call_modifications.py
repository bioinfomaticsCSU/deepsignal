"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
output format: chromosome, pos, strand, pos_in_strand, read_name, read_strand,
prob_0, prob_1, called_label, seq
"""

from __future__ import absolute_import

import tensorflow as tf
import argparse
import os
import sys
import numpy as np
from sklearn import metrics

import multiprocessing as mp
from .utils.process_utils import Queue
import time

from .model import Model
from .utils.process_utils import base2code_dna
from .utils.process_utils import code2base_dna
from .utils.process_utils import str2bool

from .extract_features import _extract_features
from .extract_features import _extract_preprocess

import uuid

queen_size_border = 2000
time_wait = 3


def _read_features_file(features_file, features_batch_q, batch_num=512):
    with open(features_file, "r") as rf:
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
                while features_batch_q.qsize() > queen_size_border:
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


def _read_features_from_fast5s(fast5s, corrected_group, basecall_subgroup, normalize_method,
                               motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
                               methy_label, batch_num=512, positions=None):
    features_list, error = _extract_features(fast5s, corrected_group, basecall_subgroup, normalize_method,
                                             motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
                                             methy_label, positions)
    features_batches = []
    for i in np.arange(0, len(features_list), batch_num):
        sampleinfo = []  # contains: chromosome, pos, strand, pos_in_strand, read_name, read_strand
        kmers = []
        base_means = []
        base_stds = []
        base_signal_lens = []
        cent_signals = []
        labels = []

        for features in features_list[i:(i + batch_num)]:
            chrom, pos, alignstrand, loc_in_ref, readname, strand, k_mer, signal_means, signal_stds, \
                signal_lens, kmer_cent_signals, f_methy_label = features

            sampleinfo.append("\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, strand]))
            kmers.append([base2code_dna[x] for x in k_mer])
            base_means.append(signal_means)
            base_stds.append(signal_stds)
            base_signal_lens.append(signal_lens)
            cent_signals.append(kmer_cent_signals)
            labels.append(f_methy_label)
        if len(sampleinfo) > 0:
            features_batches.append((sampleinfo, kmers, base_means, base_stds,
                                     base_signal_lens, cent_signals, labels))
    return features_batches, error


def _read_features_fast5s_q(fast5s_q, features_batch_q, errornum_q, corrected_group,
                            basecall_subgroup, normalize_method,
                            motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
                            methy_label, batch_num, positions):
    while True:
        if fast5s_q.empty():
            time.sleep(time_wait)
        fast5s = fast5s_q.get()
        if fast5s == "kill":
            fast5s_q.put("kill")
            break
        features_batches, error = _read_features_from_fast5s(fast5s, corrected_group, basecall_subgroup,
                                                             normalize_method, motif_seqs, methyloc,
                                                             chrom2len, kmer_len, raw_signals_len, methy_label,
                                                             batch_num, positions)
        errornum_q.put(error)
        for features_batch in features_batches:
            features_batch_q.put(features_batch)
        while features_batch_q.qsize() > queen_size_border:
            time.sleep(time_wait)


def _call_mods(features_batch, tf_sess, model, init_learning_rate):
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
    activation_logits, prediction = tf_sess.run(
        [model.activation_logits, model.prediction], feed_dict=feed_dict)
    accuracy = metrics.accuracy_score(
        y_true=labels, y_pred=prediction)

    pred_str = []
    for idx in range(labels.shape[0]):
        # chromosome, pos, strand, pos_in_strand, read_name, read_strand, prob_0, prob_1, called_label, seq
        prob_0, prob_1 = activation_logits[idx][0], activation_logits[idx][1]
        prob_0_norm = prob_0 / (prob_0 + prob_1)
        prob_1_norm = prob_1 / (prob_0 + prob_1)
        pred_str.append("\t".join([sampleinfo[idx], str(prob_0_norm),
                                   str(prob_1_norm), str(prediction[idx]),
                                   ''.join([code2base_dna[x] for x in kmers[idx]])]))

    return pred_str, accuracy


def _call_mods_q(init_learning_rate, class_num, model_path,
                 base_num, signal_num, features_batch_q, pred_str_q,
                 success_file,
                 is_rnn, is_base, is_cnn):

    model = Model(base_num=base_num,
                  signal_num=signal_num, class_num=class_num,
                  is_cnn=is_cnn, is_rnn=is_rnn, is_base=is_base)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        sess.run(tf.local_variables_initializer())

        accuracy_list = []
        count = 0
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

            pred_str, accuracy = _call_mods(features_batch, sess, model, init_learning_rate)

            pred_str_q.put(pred_str)
            accuracy_list.append(accuracy)
            count += 1
        # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
        print('call_mods process {} ending, proceed {} batches'.format(os.getpid(), count))


def _fast5s_q_to_pred_str_q(fast5s_q, errornum_q, pred_str_q,
                            corrected_group, basecall_subgroup, normalize_method,
                            motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
                            methy_label, batch_num,
                            init_learning_rate, class_num, model_path,
                            positions,
                            is_rnn, is_base, is_cnn):
    model = Model(base_num=kmer_len,
                  signal_num=raw_signals_len, class_num=class_num,
                  is_cnn=is_cnn, is_rnn=is_rnn, is_base=is_base)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        sess.run(tf.local_variables_initializer())

        accuracy_list = []
        count = 0

        while True:
            if fast5s_q.empty():
                time.sleep(time_wait)
            fast5s = fast5s_q.get()
            if fast5s == "kill":
                fast5s_q.put("kill")
                break
            features_batches, error = _read_features_from_fast5s(fast5s, corrected_group, basecall_subgroup,
                                                                 normalize_method, motif_seqs, methyloc,
                                                                 chrom2len, kmer_len, raw_signals_len, methy_label,
                                                                 batch_num, positions)
            errornum_q.put(error)
            for features_batch in features_batches:
                pred_str, accuracy = _call_mods(features_batch, sess, model, init_learning_rate)

                pred_str_q.put(pred_str)
                accuracy_list.append(accuracy)
                count += 1
        # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
        print('call_mods process {} ending, proceed {} batches'.format(os.getpid(), count))


def _write_predstr_to_file(write_fp, predstr_q):
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep()
            if predstr_q.empty():
                time.sleep(time_wait)
                continue
            pred_str = predstr_q.get()
            if pred_str == "kill":
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def _call_mods_from_fast5s_cpu(motif_seqs, chrom2len, fast5s_q, len_fast5s,
                               corrected_group, basecall_subgroup, normalize_method,
                               mod_loc, kmer_len, cent_signals_len, methy_label, batch_size,
                               learning_rate, class_num, model_path, success_file, result_file,
                               nproc, positions, is_rnn, is_base, is_cnn):

    # errornum_q = mp.Queue()
    errornum_q = Queue()

    # pred_str_q = mp.Queue()
    pred_str_q = Queue()

    if nproc > 1:
        nproc -= 1

    fast5s_q.put("kill")
    pred_str_procs = []
    for _ in range(nproc):
        p = mp.Process(target=_fast5s_q_to_pred_str_q, args=(fast5s_q, errornum_q, pred_str_q,
                                                             corrected_group, basecall_subgroup,
                                                             normalize_method, motif_seqs, mod_loc,
                                                             chrom2len, kmer_len, cent_signals_len,
                                                             methy_label, batch_size, learning_rate,
                                                             class_num, model_path, positions,
                                                             is_rnn, is_base, is_cnn))
        p.daemon = True
        p.start()
        pred_str_procs.append(p)

    print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(result_file, pred_str_q))
    p_w.daemon = True
    p_w.start()

    errornum_sum = 0
    while True:
        running = any(p.is_alive() for p in pred_str_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in pred_str_procs:
        p.join()

    print("finishing the write_process..")
    pred_str_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed.." % (errornum_sum, len_fast5s))


def _call_mods_from_fast5s_gpu(motif_seqs, chrom2len, fast5s_q, len_fast5s,
                               corrected_group, basecall_subgroup, normalize_method,
                               mod_loc, kmer_len, cent_signals_len, methy_label, batch_size,
                               learning_rate, class_num, model_path, success_file, result_file,
                               nproc, positions, is_rnn, is_base, is_cnn):
    # features_batch_q = mp.Queue()
    # errornum_q = mp.Queue()
    features_batch_q = Queue()
    errornum_q = Queue()

    # pred_str_q = mp.Queue()
    pred_str_q = Queue()

    if nproc < 2:
        nproc = 2
    elif nproc > 2:
        nproc -= 1

    fast5s_q.put("kill")
    features_batch_procs = []
    for _ in range(nproc - 1):
        p = mp.Process(target=_read_features_fast5s_q, args=(fast5s_q, features_batch_q, errornum_q,
                                                             corrected_group, basecall_subgroup,
                                                             normalize_method, motif_seqs, mod_loc,
                                                             chrom2len, kmer_len, cent_signals_len,
                                                             methy_label, batch_size, positions))
        p.daemon = True
        p.start()
        features_batch_procs.append(p)

    p_call_mods_gpu = mp.Process(target=_call_mods_q, args=(learning_rate, class_num, model_path,
                                                            kmer_len, cent_signals_len, features_batch_q,
                                                            pred_str_q, success_file, is_rnn, is_base, is_cnn))
    p_call_mods_gpu.daemon = True
    p_call_mods_gpu.start()

    print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(result_file, pred_str_q))
    p_w.daemon = True
    p_w.start()

    errornum_sum = 0
    while True:
        running = any(p.is_alive() for p in features_batch_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in features_batch_procs:
        p.join()
    features_batch_q.put("kill")

    p_call_mods_gpu.join()

    print("finishing the write_process..")
    pred_str_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed.." % (errornum_sum, len_fast5s))


def call_mods(input_path, model_path, result_file, kmer_len, cent_signals_len,
              batch_size, learning_rate, class_num, nproc, is_gpu, is_rnn, is_base, is_cnn,
              f5_args):
    start = time.time()

    model_path = os.path.abspath(model_path)
    input_path = os.path.abspath(input_path)
    success_file = input_path.rstrip("/") + "." + str(uuid.uuid1()) + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)

    if os.path.isdir(input_path):
        is_recursive, corrected_group, basecall_subgroup, reference_path, is_dna, \
            normalize_method, motifs, mod_loc, methy_label, f5_batch_num, position_file = f5_args

        motif_seqs, chrom2len, fast5s_q, len_fast5s, positions = _extract_preprocess(input_path, is_recursive,
                                                                                     motifs, is_dna,
                                                                                     reference_path, f5_batch_num,
                                                                                     position_file)

        if is_gpu:
            _call_mods_from_fast5s_gpu(motif_seqs, chrom2len, fast5s_q, len_fast5s, corrected_group,
                                       basecall_subgroup, normalize_method, mod_loc, kmer_len, cent_signals_len,
                                       methy_label, batch_size, learning_rate, class_num, model_path, success_file,
                                       result_file, nproc, positions,
                                       is_rnn, is_base, is_cnn)
        else:
            _call_mods_from_fast5s_cpu(motif_seqs, chrom2len, fast5s_q, len_fast5s, corrected_group,
                                       basecall_subgroup, normalize_method, mod_loc, kmer_len, cent_signals_len,
                                       methy_label, batch_size, learning_rate, class_num, model_path, success_file,
                                       result_file, nproc, positions,
                                       is_rnn, is_base, is_cnn)

    else:
        # features_batch_q = mp.Queue()
        features_batch_q = Queue()
        p_rf = mp.Process(target=_read_features_file, args=(input_path, features_batch_q, batch_size))
        p_rf.daemon = True
        p_rf.start()

        # pred_str_q = mp.Queue()
        pred_str_q = Queue()

        predstr_procs = []
        if nproc > 2:
            nproc -= 2

        if is_gpu:
            nproc_tf = 1
        else:
            nproc_tf = nproc

        for _ in range(nproc_tf):
            p = mp.Process(target=_call_mods_q, args=(learning_rate, class_num, model_path,
                                                      kmer_len, cent_signals_len, features_batch_q,
                                                      pred_str_q, success_file, is_rnn, is_base, is_cnn))
            p.daemon = True
            p.start()
            predstr_procs.append(p)

        print("write_process started..")
        p_w = mp.Process(target=_write_predstr_to_file, args=(result_file, pred_str_q))
        p_w.daemon = True
        p_w.start()

        for p in predstr_procs:
            p.join()

        print("finishing the write_process..")
        pred_str_q.put("kill")

        p_rf.join()

        p_w.join()

    if os.path.exists(success_file):
        os.remove(success_file)
    print("call_mods costs %.2f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input_path", "-i", action="store", type=str,
                         required=True,
                         help="the input path, can be a signal_feature file from extract_features.py, "
                              "or a directory of fast5 files. If a directory of fast5 files is provided, "
                              "args in FAST5_EXTRACTION should (reference_path must) be provided.")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_path", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")

    p_call.add_argument('--is_cnn', type=str, default='yes', required=False,
                        help="dose the used model contain inception module?")
    p_call.add_argument('--is_rnn', type=str, default='yes', required=False,
                        help="dose the used model contain BiLSTM module?")
    p_call.add_argument('--is_base', type=str, default='yes', required=False,
                        help="dose the BiLSTM module of the used model take base features as input?")

    p_call.add_argument("--kmer_len", "-x", action="store", default=17, type=int, required=False,
                        help="base num of the kmer, default 17")
    p_call.add_argument("--cent_signals_len", "-y", action="store", default=360, type=int, required=False,
                        help="the number of central signals of the kmer to be used, default 360")
    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")
    p_call.add_argument("--learning_rate", "-l", default=0.001, type=float, required=False,
                        action="store", help="init learning rate, default 0.001")
    p_call.add_argument("--class_num", "-c", action="store", default=2, type=int, required=False,
                        help="class num, default 2")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--result_file", "-o", action="store", type=str, required=True,
                          help="the file path to save the predicted result")

    p_f5 = parser.add_argument_group("FAST5_EXTRACTION")
    p_f5.add_argument("--recursively", "-r", action="store", type=str, required=False,
                      default='yes', help='is to find fast5 files from fast5 dir recursively. '
                                          'default true, t, yes, 1')
    p_f5.add_argument("--corrected_group", action="store", type=str, required=False,
                      default='RawGenomeCorrected_000',
                      help='the corrected_group of fast5 files after '
                           'tombo re-squiggle. default RawGenomeCorrected_000')
    p_f5.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                      default='BaseCalled_template',
                      help='the corrected subgroup of fast5 files. default BaseCalled_template')
    p_f5.add_argument("--reference_path", action="store",
                      type=str, required=False,
                      help="the reference file to be used, usually is a .fa file")
    p_f5.add_argument("--is_dna", action="store", type=str, required=False,
                      default='yes',
                      help='whether the fast5 files from DNA sample or not. '
                           'default true, t, yes, 1. '
                           'setting this option to no/false/0 means '
                           'the fast5 files are from RNA sample.')
    p_f5.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                      default="mad", required=False,
                      help="the way for normalizing signals in read level. "
                           "mad or zscore, default mad")
    # p_f5.add_argument("--methy_label", action="store", type=int,
    #                   choices=[1, 0], required=False, default=1,
    #                   help="the label of the interested modified bases, this is for training."
    #                        " 0 or 1, default 1")
    p_f5.add_argument("--motifs", action="store", type=str,
                      required=False, default='CG',
                      help='motif seq to be extracted, default: CG. '
                           'can be multi motifs splited by comma '
                           '(no space allowed in the input str), '
                           'or use IUPAC alphabet, '
                           'the mod_loc of all motifs must be '
                           'the same')
    p_f5.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                      help='0-based location of the targeted base in the motif, default 0')
    p_f5.add_argument("--f5_batch_num", action="store", type=int, default=100,
                      required=False,
                      help="number of files to be processed by each process one time, default 100")
    p_f5.add_argument("--positions", action="store", type=str,
                      required=False, default=None,
                      help="file with a list of positions interested (must be formatted as tab-separated file"
                           " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                           "need to be set. --positions is used to narrow down the range of the trageted "
                           "motif locs. default None")

    parser.add_argument("--nproc", "-p", action="store", type=int, default=1,
                        required=False, help="number of processes to be used, default 1.")
    parser.add_argument("--is_gpu", action="store", type=str, default="no", required=False,
                        choices=["yes", "no"], help="use gpu for tensorflow or not, default no. "
                                                    "If you're using a gpu machine, please set to yes. "
                                                    "Note that when is_gpu is yes, --nproc is not valid "
                                                    "to tensorflow.")

    args = parser.parse_args()

    input_path = args.input_path

    model_path = args.model_path
    result_file = args.result_file

    is_cnn = str2bool(args.is_cnn)
    is_base = str2bool(args.is_base)
    is_rnn = str2bool(args.is_rnn)

    kmer_len = args.kmer_len
    cent_signals_len = args.cent_signals_len

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    class_num = args.class_num

    nproc = args.nproc
    is_gpu = str2bool(args.is_gpu)

    # for FAST5_EXTRACTION
    is_recursive = str2bool(args.recursively)
    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    reference_path = args.reference_path
    is_dna = str2bool(args.is_dna)
    normalize_method = args.normalize_method
    motifs = args.motifs
    mod_loc = args.mod_loc
    # methy_label = args.methy_label
    methy_label = 1
    f5_batch_num = args.f5_batch_num
    position_file = args.positions

    f5_args = (is_recursive, corrected_group, basecall_subgroup, reference_path, is_dna,
               normalize_method, motifs, mod_loc, methy_label, f5_batch_num, position_file)

    call_mods(input_path, model_path, result_file, kmer_len, cent_signals_len,
              batch_size, learning_rate, class_num, nproc, is_gpu, is_rnn, is_base, is_cnn,
              f5_args)


if __name__ == '__main__':
    sys.exit(main())
