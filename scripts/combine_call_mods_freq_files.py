#! /usr/bin/python
import argparse


def _read_one_mod_freq_file(freqfile):
    freqinfo = {}
    with open(freqfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            m_key = (words[0], int(words[1]), words[2])
            pos_in_strand = words[3]
            methy_prob = float(words[4])
            unmethy_prob = float(words[5])
            methy_cov = int(words[6])
            unmethy_cov = int(words[7])
            cov = int(words[8])
            rmet = float(words[9])
            kmer = words[10]
            freqinfo[m_key] = [pos_in_strand, methy_prob, unmethy_prob, methy_cov, unmethy_cov,
                               cov, rmet, kmer]
    return freqinfo


def _get_combined_freq_file(freqfiles):
    freqinfo = {}
    freqkeys = set()
    for ffile in freqfiles:
        finfo_tmp = _read_one_mod_freq_file(ffile)
        for fkey in finfo_tmp.keys():
            if fkey not in freqkeys:
                freqkeys.add(fkey)
                freqinfo[fkey] = ["", 0.0, 0.0, 0, 0, 0, 0.0, ""]
            freqinfo[fkey][0] = finfo_tmp[fkey][0]
            freqinfo[fkey][1] += finfo_tmp[fkey][1]
            freqinfo[fkey][2] += finfo_tmp[fkey][2]
            freqinfo[fkey][3] += finfo_tmp[fkey][3]
            freqinfo[fkey][4] += finfo_tmp[fkey][4]
            freqinfo[fkey][5] += finfo_tmp[fkey][5]
            freqinfo[fkey][6] = freqinfo[fkey][3] / float(freqinfo[fkey][5])
            freqinfo[fkey][7] = finfo_tmp[fkey][7]
    return freqinfo


def _write_freqinfo(freqinfo, wfile):
    wf = open(wfile, "w")
    fkeys = sorted(list(freqinfo.keys()))
    for fkey in fkeys:
        tmpinfo = list(fkey) + freqinfo[fkey]
        wf.write("%s\t%d\t%s\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%.3f\t%s\n" % (tmpinfo[0], tmpinfo[1], tmpinfo[2],
                                                                                 tmpinfo[3],
                                                                                 tmpinfo[4],
                                                                                 tmpinfo[5],
                                                                                 tmpinfo[6], tmpinfo[7],
                                                                                 tmpinfo[8], tmpinfo[9],
                                                                                 tmpinfo[10]))
    wf.close()


def combine_freq_files(args):
    modsfiles = args.modsfile
    freqinfo = _get_combined_freq_file(modsfiles)
    _write_freqinfo(freqinfo, args.wfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modsfile", action="append", type=str, required=True,
                        help="call_mods_freq file")
    parser.add_argument("--wfile", type=str, required=True,
                        help="")

    args = parser.parse_args()
    combine_freq_files(args)


if __name__ == '__main__':
    main()
