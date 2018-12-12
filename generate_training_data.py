# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     generate_training_data
   Description :
   Author :       huangneng
   date：          2018/8/6
-------------------------------------------------
   Change Activity:
                   2018/8/6:
-------------------------------------------------
"""
__author__ = 'huangneng'

import os
import shutil
import struct
import argparse

DNA_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def extract(filename, out_dir, format_string, max_rname_len):
    fmt = format_string
    fread = open(filename, 'r')
    bin_writer = open(out_dir + os.path.sep + 'training.bin', 'wb')
    for line in fread:
        name, base_char, means, stds, siglen, signals, label = line.strip().split("\t")
        name_str = name + ' ' * (max_rname_len - len(name))
        base_int = [DNA_BASE[v] for v in base_char]
        means = [float(v) for v in means.split(',')]
        stds = [float(v) for v in stds.split(',')]
        siglen = [int(v) for v in siglen.split(',')]
        signals = [float(v) for v in signals.split(',')]
        label = int(label)
        bin_writer.write(struct.pack(fmt, *base_int + means + stds + siglen + signals + [label], str.encode(name_str)))
    fread.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o', '--out_dir', default='./training/')
    parser.add_argument('-m', '--max_rname_len', default=100, type=int)
    parser.add_argument('-b', '--base_num', default=17, type=int)
    parser.add_argument('-s', '--signal_num', default=120, type=int)
    argv = parser.parse_args()
    if os.path.exists(argv.out_dir):
        print('deleting the previous output direction...')
        shutil.rmtree(argv.out_dir)
        print('done')
    os.makedirs(argv.out_dir)
    format_string = '<' + str(argv.base_num) + 'B' + str(argv.base_num) + 'f' + str(argv.base_num) + 'f' + str(
        argv.base_num) + 'H' + str(argv.signal_num) + 'f' + '1B' + str(argv.max_rname_len) + 's'
    extract(argv.input_file, argv.out_dir, format_string, argv.max_rname_len)
