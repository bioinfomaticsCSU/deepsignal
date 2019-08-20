# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     generate_binary_feature_file
   Description : convert txt file (from extract_features module) to binary format
   Author :       huangneng, nipeng
   date：          2018/8/6
-------------------------------------------------
   Change Activity:
                   2018/8/6:
-------------------------------------------------
"""
__author__ = 'huangneng'

import os
import struct
import argparse

DNA_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def extract(filename, out_file, format_string):
    fmt = format_string
    fread = open(filename, 'r')
    if out_file is None:
        fname, fext = os.path.splitext(filename)
        out_file = fname + ".bin"
    bin_writer = open(out_file, 'wb')
    for line in fread:
        words = line.strip().split("\t")
        base_char, means, stds, siglen, signals, label = words[6], words[7], words[8], words[9], words[10], words[11]
        base_int = [DNA_BASE[v] for v in base_char]
        means = [float(v) for v in means.split(',')]
        stds = [float(v) for v in stds.split(',')]
        siglen = [int(v) for v in siglen.split(',')]
        signals = [float(v) for v in signals.split(',')]
        label = int(label)
        bin_writer.write(struct.pack(fmt, *base_int + means + stds + siglen + signals + [label]))
    fread.close()
    bin_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o', '--out_file', default=None, required=False)
    parser.add_argument('-b', '--base_num', default=17, type=int)
    parser.add_argument('-s', '--signal_num', default=360, type=int)
    argv = parser.parse_args()

    format_string = '<' + str(argv.base_num) + 'B' + str(argv.base_num) + 'f' + str(argv.base_num) + 'f' + str(
        argv.base_num) + 'H' + str(argv.signal_num) + 'f' + '1B'
    extract(argv.input_file, argv.out_file, format_string)


if __name__ == '__main__':
    main()
