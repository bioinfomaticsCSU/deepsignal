#!/usr/bin/python
import sys
import os
import math
import random
import argparse


def _read_kmer2ratio_file(krfile):
    kmer2ratio = {}
    with open(krfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            kmer2ratio[words[0]] = float(words[2])
    return kmer2ratio


def _get_kmer2lines(feafile):
    kmer2lines = {}
    kmers = set()
    with open(feafile, "r") as rf:
        lcnt = 0
        for line in rf:
            words = line.strip().split("\t")
            kmer = words[6]
            if kmer not in kmers:
                kmers.add(kmer)
                kmer2lines[kmer] = []
            kmer2lines[kmer].append(lcnt)
            lcnt += 1
    return kmer2lines


def _rand_select_by_kmer_ratio(kmer2lines, kmer2ratios, totalline):
    selected_lines = []
    unratioed_kmers = set()
    cnts = 0
    for kmer in kmer2lines.keys():
        if kmer in kmer2ratios.keys():
            linenum = int(math.ceil(totalline * kmer2ratios[kmer]))
            lines = kmer2lines[kmer]
            if len(lines) <= linenum:
                selected_lines += lines
                cnts += (linenum - len(lines))
            else:
                selected_lines += random.sample(lines, linenum)
        else:
            unratioed_kmers.add(kmer)
    print("for {} common kmers, fill {} samples, "
          "{} samples that can't filled".format(len(kmer2lines.keys()) - len(unratioed_kmers),
                                                len(selected_lines),
                                                cnts))
    unfilled_cnt = totalline - len(selected_lines)
    print("totalline: {}, need to fill: {}".format(totalline, unfilled_cnt))
    if len(unratioed_kmers) > 0:
        minlinenum = int(math.ceil(float(unfilled_cnt)/len(unratioed_kmers)))
        cnts = 0
        for kmer in unratioed_kmers:
            lines = kmer2lines[kmer]
            if len(lines) <= minlinenum:
                selected_lines += lines
                cnts += len(lines)
            else:
                selected_lines += random.sample(lines, minlinenum)
                cnts += minlinenum
        print("extract {} samples from {} diff kmers".format(cnts, len(unratioed_kmers)))
    selected_lines = sorted(selected_lines)
    selected_lines = [-1] + selected_lines
    return selected_lines


def _write_randsel_lines(feafile, wfile, seled_lines):
    wf = open(wfile, 'w')
    with open(feafile) as rf:
        for i in range(1, len(seled_lines)):
            chosen_line = ''
            for j in range(0, seled_lines[i] - seled_lines[i - 1]):
                # print(j)
                chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    print('random_select_file_rows finished..')


def select_neg_samples_by_kmer_disti(args):
    kmer2ratio = _read_kmer2ratio_file(args.krfile)
    print("{} kmers from kmer2ratio file".format(len(kmer2ratio)))
    kmer2lines = _get_kmer2lines(args.feafile)
    sel_lines = _rand_select_by_kmer_ratio(kmer2lines, kmer2ratio, args.totalline)
    _write_randsel_lines(args.feafile, args.wfile, sel_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feafile", type=str)
    parser.add_argument("--krfile", type=str, help="kmer2ratio file")
    parser.add_argument("--totalline", type=int, help="")
    parser.add_argument("--wfile", type=str)

    args = parser.parse_args()
    select_neg_samples_by_kmer_disti(args)


if __name__ == '__main__':
    main()
