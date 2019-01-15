#! /usr/bin/env python
"""
calculate modification frequency at genome level
"""

import argparse
import os
import sys

from txt_formater import ModRecord
from txt_formater import SiteStats
from txt_formater import split_key


def calculate_mods_frequency(mods_files, prob_cf):
    sitekeys = set()
    sitekey2stats = dict()

    count, used = 0, 0
    for mods_file in mods_files:
        with open(mods_file, 'r') as rf:
            for line in rf:
                words = line.strip().split("\t")
                mod_record = ModRecord(words)
                if mod_record.is_record_callable(prob_cf):
                    if mod_record._site_key not in sitekeys:
                        sitekeys.add(mod_record._site_key)
                        sitekey2stats[mod_record._site_key] = SiteStats(mod_record._strand,
                                                                        mod_record._pos_in_strand,
                                                                        mod_record._kmer)
                    sitekey2stats[mod_record._site_key]._prob_0 += mod_record._prob_0
                    sitekey2stats[mod_record._site_key]._prob_1 += mod_record._prob_1
                    sitekey2stats[mod_record._site_key]._coverage += 1
                    if mod_record._called_label == 1:
                        sitekey2stats[mod_record._site_key]._met += 1
                    else:
                        sitekey2stats[mod_record._site_key]._unmet += 1
                    used += 1
                count += 1
    print("{} of {} samples used..".format(used, count))
    return sitekey2stats


def write_sitekey2stats(sitekey2stats, result_file):
    sorted_keys = sorted(list(sitekey2stats.keys()), key=lambda x: split_key(x))

    with open(result_file, 'w') as wf:
        # wf.write('\t'.join(['chromosome', 'pos', 'strand', 'pos_in_strand', 'prob0', 'prob1',
        #                     'met', 'unmet', 'coverage', 'Rmet', 'kmer']) + '\n')
        for key in sorted_keys:
            chrom, pos = split_key(key)
            sitestats = sitekey2stats[key]
            assert(sitestats._coverage == (sitestats._met + sitestats._unmet))
            if sitestats._coverage > 0:
                rmet = float(sitestats._met) / sitestats._coverage
                wf.write("%s\t%d\t%s\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%.3f\t%s\n" % (chrom, pos, sitestats._strand,
                                                                                 sitestats._pos_in_strand,
                                                                                 sitestats._prob_0,
                                                                                 sitestats._prob_1,
                                                                                 sitestats._met, sitestats._unmet,
                                                                                 sitestats._coverage, rmet,
                                                                                 sitestats._kmer))
            else:
                print("{} {} has no coverage..".format(chrom, pos))


def main():
    parser = argparse.ArgumentParser(description='calculate frequency of interested sites at genome level')
    parser.add_argument('--input_path', '-i', action="store", type=str, required=True,
                        help='a result file from call_modifications.py, or a directory contains a bunch of '
                             'result files.')
    parser.add_argument('--result_file', '-o', action="store", type=str, required=True,
                        help='the file path to save the result')
    parser.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                        help='this is to remove ambiguous calls. '
                             'if abs(prob1-prob0)>=prob_cf, then we use the call. '
                             'range 0-1, default 0.')
    parser.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                        help='a unique str which all input files has, this is for find all input files  and ignore '
                             'the un-input-files in a input directory. if input_path is a file, ignore this arg.')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    result_file = args.result_file
    prob_cf= args.prob_cf
    file_uid = args.file_uid

    mods_files = []
    if os.path.isdir(input_path):
        for ifile in os.listdir(input_path):
            if file_uid is None:
                mods_files.append('/'.join([input_path, ifile]))
            elif ifile.find(file_uid) != -1:
                mods_files.append('/'.join([input_path, ifile]))
    elif os.path.isfile(input_path):
        mods_files.append(input_path)
    else:
        raise ValueError()
    print("get {} input file(s)..".format(len(mods_files)))

    print("reading the input files..")
    sites_stats = calculate_mods_frequency(mods_files, prob_cf)
    print("writing the result..")
    write_sitekey2stats(sites_stats, result_file)


if __name__ == '__main__':
    sys.exit(main())
