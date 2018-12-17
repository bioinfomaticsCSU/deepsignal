#! /usr/bin/python
"""
id  signals signal2base_loc signal2base_base   centbaseloc maskinfo    label
signal2base_loc, signal2base_base: the mapping of signal arrays to bases
centbaseloc: central base loc info in the trimed signal array
"""
import argparse
import numpy as np
import time
import os
import random
from _utils import max_num_bases
from _utils import str2bool
from _ref_reader import get_contig2len

MAX_BASE_NUM = max_num_bases()
MAX_LEGAL_SIGNAL_NUM = 800  # for now 800 only for baseseq with len of 17
MAX_LEGAL_SIGNAL_NUM_ONE_BASE = 40
is_sampling = True
base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def parse_k_mer_signals(k_mer, k_signals,
                        is_only_hq_samples, cut_per_side, rawsignal_num):
    cent_bases = k_mer
    signals_list = [x.split(',') for x in k_signals.strip().split(';')]
    signals_list = [[float(x) for x in y] for y in signals_list]

    if cut_per_side != 0:
        cent_bases = cent_bases[cut_per_side:(-cut_per_side)]
        signals_list = signals_list[cut_per_side:(-cut_per_side)]
    signal_lens = [len(x) for x in signals_list]
    if is_only_hq_samples and sum(signal_lens) > MAX_LEGAL_SIGNAL_NUM:
        return None, None, None, None, None

    # if is_sampling and sum(signal_lens) > rawsignal_num:
    #     for i in range(0, len(signal_lens)):
    #         if signal_lens[i] > MAX_LEGAL_SIGNAL_NUM_ONE_BASE:
    #             signal_per_base_tmp = signals_list[i]
    #             signals_list[i] = [signal_per_base_tmp[x] for x in sorted(
    #                 random.sample(range(signal_lens[i]), MAX_LEGAL_SIGNAL_NUM_ONE_BASE))]
    #     signal_lens = [len(x) for x in signals_list]
    signal_means = [np.mean(x) for x in signals_list]
    signal_stds = [np.std(x) for x in signals_list]

    if sum(signal_lens) < rawsignal_num:
        real_signals = sum(signals_list, [])
        cent_signals = real_signals + [0] * (rawsignal_num - len(real_signals))
    else:
        mid_loc = int((len(signals_list) - 1) / 2)
        mid_base_len = len(signals_list[mid_loc])

        if mid_base_len >= rawsignal_num:
            allcentsignals = signals_list[mid_loc]
            cent_signals = [allcentsignals[x] for x in sorted(random.sample(range(len(allcentsignals)),
                                                                            rawsignal_num))]
        else:
            left_len = (rawsignal_num - mid_base_len) // 2
            right_len = rawsignal_num - left_len

            left_signals = sum(signals_list[:mid_loc], [])
            right_signals = sum(signals_list[mid_loc:], [])

            if left_len > len(left_signals):
                right_len = right_len + left_len - len(left_signals)
                left_len = len(left_signals)
            elif right_len > len(right_signals):
                left_len = left_len + right_len - len(right_signals)
                right_len = len(right_signals)

            assert (right_len + left_len == rawsignal_num)
            if left_len == 0:
                cent_signals = right_signals[:right_len]
            else:
                cent_signals = left_signals[-left_len:] + right_signals[:right_len]

    return cent_bases, signal_means, signal_stds, signal_lens, cent_signals


def parse_one_line(line, is_only_hq_samples, cut_per_side=0, rawsignal_num=120):
    """
    :param line: a line from cpg_signal_features.tsv
    (genereted by make_inputs)
    :param is_only_hq_samples:
    :param cut_per_side:
    :param rawsignal_num:
    :return: id, features (list), mask (list), label
    masK: two elements, number of 0s of two sides
    """
    words = line.strip().split('\t')
    sample_id = words[0]
    label = int(words[1])
    k_mer = words[-2]
    k_signals = words[-1]

    cent_bases, signal_means, signal_stds, signal_lens, cent_signals = parse_k_mer_signals(k_mer, k_signals,
                                                                                           is_only_hq_samples,
                                                                                           cut_per_side,
                                                                                           rawsignal_num)

    return sample_id, cent_bases, signal_means, signal_stds, signal_lens, cent_signals, label


def _parse_sampleid(sampleid):
    """
    7f56bcc8-ebac-41be-bf83-fa1d9938c1a5chr16+42203t
    :param sampleid:
    :return:
    """
    readid = sampleid[0:36]
    alignmentinfo_str = sampleid[36:-1]
    readstrand = sampleid[-1]
    chromsome, strand, loc = '', '', -1
    if '+' in alignmentinfo_str:
        strand = '+'
        candl = alignmentinfo_str.split('+')
        chromsome, loc = candl[0], int(candl[1])
    else:
        strand = '-'
        candl = alignmentinfo_str.split('-')
        chromsome, loc = candl[0], int(candl[1])
    return readid, chromsome, strand, loc, readstrand


def modify_sampleid(sampleid, contig2length):
    readid, chromsome, strand, loc, readstrand = _parse_sampleid(sampleid)
    if strand == '-':
        reverse_cpg_loc = (contig2length[chromsome] - 1 - loc) - 1
        rc_sampleid = ''.join([readid, chromsome, '+', str(reverse_cpg_loc), readstrand])
        return rc_sampleid
    else:
        return sampleid


def read_methy_signal_features_file(methy_signal_features_file,
                                    header,
                                    w_filepath,
                                    is_norm_by_read, is_moid,
                                    is_only_hq_samples,
                                    contig2len,
                                    base_num=17,
                                    rawsignals_num=120):
    if 1 > base_num or base_num > MAX_BASE_NUM or base_num % 2 == 0:
        raise ValueError("base_num must be odd, 1<=base_num<=MAX_BASE_NUM")

    cut_per_side = 0
    if base_num != MAX_BASE_NUM:
        cut_per_side = (MAX_BASE_NUM - base_num) // 2

    # sample_ids, features_list, mask_list, labels = [], [], [], []
    wf = open(w_filepath, 'w')
    with open(methy_signal_features_file, 'r') as rf:
        if header:
            next(rf)
        for line in rf:
            sample_id, nu_bases, signal_means, signal_stds, signal_lens, cent_signals, \
                label = parse_one_line(line, is_only_hq_samples, cut_per_side, rawsignals_num)
            if signal_means is not None:
                means_text = ','.join([str(x) for x in signal_means])
                stds_text = ','.join([str(x) for x in signal_stds])
                signal_len_text = ','.join([str(x) for x in signal_lens])
                cent_signals_text = ','.join([str(x) for x in cent_signals])

                if is_moid:
                    sample_id = modify_sampleid(sample_id, contig2len)

                wf.write('\t'.join([sample_id, nu_bases, means_text,
                                    stds_text, signal_len_text, cent_signals_text,
                                    str(label)]) + '\n')
    wf.flush()
    wf.close()


def _replace_baseinfo(filename, basesnum):
    # FIXME: 2, 7 is not safe
    bases_loc = filename.find('bases')
    if bases_loc != -1:
        startloc = bases_loc - 2
        return filename.replace(filename[startloc:startloc + 7], str(basesnum)+'bases')
    return filename + '.17bases'


def main():
    parser = argparse.ArgumentParser(description='extract info from signal_features_file generated '
                                                 'by extract_kmer_signals.py')
    parser.add_argument('--sf_filepath', type=str, required=True,
                        help='the signal_features_file path')
    parser.add_argument('--header', type=str, default='yes', required=False,
                        help='whether there are headers in sf_filepath or not. default yes')
    parser.add_argument('--bases_num', type=int, default=17, required=False,
                        help='bases_num to extract signals, default 17')
    parser.add_argument('--raw_signals_num', type=int, default=375, required=False,
                        help='number of signals to extract around the central base. default 120')
    # for modifying ID
    parser.add_argument('--if_modi_ID', type=str, default='no', required=False,
                        help='if modi the alignment pos in the sample IDs or not. default no')
    parser.add_argument('--ref_fa', type=str, required=False,
                        default=None,
                        help='ref fa file path')
    # for normalize feature signals by reads
    parser.add_argument('--if_normalize_by_reads', type=str, default='no', required=False,
                        help='if do the feature normalization by reads or not. default no')
    # for filter samples by signal quality (signal length etc.)
    parser.add_argument('--if_filter_samples', type=str, default='no', required=False,
                        help='if filter samples by signals quality (signal length etc.). default no')
    args = parser.parse_args()

    sf_fp = args.sf_filepath
    header = str2bool(args.header)
    bases_num = args.bases_num
    rawsignals_num = args.raw_signals_num

    ismoid = str2bool(args.if_modi_ID)
    ref_fp = args.ref_fa

    is_norm_read = str2bool(args.if_normalize_by_reads)

    is_filter_samples = str2bool(args.if_filter_samples)

    fname, fext = os.path.splitext(sf_fp)
    midfix = ''
    midfix += ".rawsignals_" + str(rawsignals_num)
    if is_norm_read:
        midfix += '.norm_read'
    if is_filter_samples:
        midfix += '.hq'
    if ismoid:
        midfix += '.moID'
    fname = _replace_baseinfo(fname, bases_num)
    w_fp = fname + midfix + fext

    start = time.time()
    chrom2len = {}
    if ismoid:
        chrom2len = get_contig2len(ref_fp)
    features_info = read_methy_signal_features_file(sf_fp, header, w_fp,
                                                    is_norm_read, ismoid,
                                                    is_filter_samples,
                                                    chrom2len, bases_num,
                                                    rawsignals_num)
    print('done in {} seconds..'.format(time.time() - start))
    pass


if __name__ == '__main__':
    main()
