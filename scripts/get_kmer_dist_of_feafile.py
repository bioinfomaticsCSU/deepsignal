#!/usr/bin/python
import sys
import os
import argparse


def _count_kmers_of_feafile(feafile):
    kmer_count = {}
    kmers = set()
    with open(feafile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            kmer = words[6]
            if kmer not in kmers:
                kmers.add(kmer)
                kmer_count[kmer] = 0
            kmer_count[kmer] += 1
    return kmer_count


def _get_3mer_ratio(kmer_count, kmer_len):
    kmer3_2_poses = {}
    start = kmer_len // 2
    for kmer in kmer_count.keys():
        kmer3 = kmer[start:(start + 3)]
        if kmer3 not in kmer3_2_poses.keys():
            kmer3_2_poses[kmer3] = 0
        kmer3_2_poses[kmer3] += kmer_count[kmer]

    poesenum = sum(list(kmer3_2_poses.values()))
    kmer3keys = sorted(list(kmer3_2_poses.keys()))
    y = []
    for kmer3 in kmer3keys:
        y.append(kmer3_2_poses[kmer3] / float(poesenum))
    print(kmer3keys, y)


def _get_kmer_ratio(kmer_count):
    total_cnt = sum(list(kmer_count.values()))
    kmer_ratios = []
    for kmer in kmer_count.keys():
        kmer_ratios.append((kmer, kmer_count[kmer], float(kmer_count[kmer])/total_cnt))
    kmer_ratios = sorted(kmer_ratios, key=lambda x: x[1], reverse=True)
    return kmer_ratios


def _write_kmer_ratio(kmer_ratio, wfile):
    wf = open(wfile, 'w')
    for kmerr in kmer_ratio:
        kmerstr = "\t".join(list(map(str, kmerr)))
        wf.write(kmerstr + "\n")
    wf.flush()
    wf.close()


def get_kmer_dist(args):
    kmer2count = _count_kmers_of_feafile(args.feafile)
    _get_3mer_ratio(kmer2count, len(list(kmer2count.keys())[0]))

    kmerratios = _get_kmer_ratio(kmer2count)

    fname, fext = os.path.splitext(args.feafile)
    ratiofile = fname + ".kmer_distri" + fext

    _write_kmer_ratio(kmerratios, ratiofile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feafile", type=str, required=True, help="")

    args = parser.parse_args()

    get_kmer_dist(args)


if __name__ == '__main__':
    main()
