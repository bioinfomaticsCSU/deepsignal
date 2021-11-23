#! /usr/bin/env python
import argparse
import os


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def filter_one_signal_feature_file(sf_fp, wfp, label):
    wf = open(wfp, 'w')
    with open(sf_fp, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            if words[-1] == label:
                wf.write(line)
    wf.close()


def filter_one_signal_feature_file_append(sf_fp, wfp, label):
    wf = open(wfp, 'a')
    with open(sf_fp, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            if words[-1] == label:
                wf.write(line)
    wf.close()
# ==========================================================================


def main():
    parser = argparse.ArgumentParser(description='extract samples with interested ref_positions '
                                                 'from signal feature file')
    parser.add_argument('--sf_path', type=str, required=True,
                        help='the sample signal_features_file path needed to be filtered, or a directory')
    parser.add_argument('--unique_fid', type=str, required=False,
                        default='.tsv',
                        help='unique str of all to be processed files')
    parser.add_argument('--midfix', type=str, required=False,
                        default="filtered",
                        help='a str to id the output file')
    parser.add_argument('--label', type=str, required=False,
                        default="1", choices=["0", "1"],
                        help="methy label of the extracted samples")

    args = parser.parse_args()
    sf_fp = args.sf_path  # signal features
    unique_fid = args.unique_fid
    midfix = args.midfix
    label = args.label

    if os.path.isdir(sf_fp):
        wfp = sf_fp.strip('/') + '.' + midfix + '.tsv'
        wf = open(wfp, 'w')
        wf.close()
        for sfile in os.listdir(sf_fp):
            if sfile.find(unique_fid) != -1:
                read_file = sf_fp + '/' + sfile
                filter_one_signal_feature_file_append(read_file, wfp, label)
                print('done with file {}'.format(read_file))
    else:
        fname, fext = os.path.splitext(sf_fp)
        wfp = fname + '.' + midfix + fext
        filter_one_signal_feature_file(sf_fp, wfp, label)


if __name__ == '__main__':
    main()
