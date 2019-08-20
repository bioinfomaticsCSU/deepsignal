#! /usr/bin/env python
"""
calculate accuracy (with other criteria) of called mods from call_modifications.py.
need two result file, one contains calls for methylated sites, another contains calls for
unmethylated sites.
"""

import argparse
import sys
import os
import random
import numpy
from collections import namedtuple
from sklearn.metrics import roc_auc_score

from txt_formater import ModRecord


num_sites = [100000, ]
prob_cfs = numpy.arange(0, 0.70, 0.025)
CallRecord = namedtuple('CallRecord', ['key', 'predicted_label', 'is_true_methylated',
                                       'prob0', 'prob1'])


def sample_sites(filename, is_methylated):
    all_crs = list()
    rf = open(filename)
    for line in rf:
        mt_record = ModRecord(line.rstrip().split())
        all_crs.append(CallRecord(mt_record._site_key, mt_record._called_label,
                                  is_methylated, mt_record._prob_0,
                                  mt_record._prob_1))
    rf.close()
    print('there are {} basemod candidates totally'.format(len(all_crs)))

    random.shuffle(all_crs)
    return all_crs


def _evaluate_(tested_sites, prob_cf):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    called = 0
    correct = 0

    y_truelabel = []
    y_scores = []

    for s in tested_sites:
        tp += s.predicted_label and s.is_true_methylated
        fp += s.predicted_label and not s.is_true_methylated
        tn += not s.predicted_label and not s.is_true_methylated
        fn += not s.predicted_label and s.is_true_methylated

        y_truelabel.append(s.is_true_methylated)
        y_scores.append(s.prob1)

        prob1_minus_prob0 = s.prob1 - s.prob0
        if abs(prob1_minus_prob0) >= prob_cf:
            called += 1
            if (prob1_minus_prob0 >= prob_cf) == s.is_true_methylated:
                correct += 1

    print(tp, fp, tn, fn)
    precision, recall, specificity, accuracy = 0, 0, 0, 0
    fall_out, miss_rate, fdr, npv = 0, 0, 0, 0
    auroc = 0
    called_accuracy = 0
    if len(tested_sites) > 0:
        accuracy = float(tp + tn) / len(tested_sites)
        if tp + fp > 0:
            precision = float(tp) / (tp + fp)
            fdr = float(fp) / (tp + fp)  # false discovery rate
        else:
            precision = 0
            fdr = 0
        if tp + fn > 0:
            recall = float(tp) / (tp + fn)
            miss_rate = float(fn) / (tp + fn)  # false negative rate
        else:
            recall = 0
            miss_rate = 0
        if tn + fp > 0:
            specificity = float(tn) / (tn + fp)
            fall_out = float(fp) / (fp + tn)  # false positive rate
        else:
            specificity = 0
            fall_out = 0
        if tn + fn > 0:
            npv = float(tn) / (tn + fn)  # negative predictive value
        else:
            npv = 0
        if called > 0:
            called_accuracy = float(correct) / called
        else:
            called_accuracy = 0
        try:
            auroc = roc_auc_score(numpy.array(y_truelabel), numpy.array(y_scores))
        except ValueError:
            # for only one kind of label
            auroc = 0
    return "%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d" \
           "\t%d\t%.3f\t%.3f" % (tp, fp, tn, fn,
                                 accuracy, recall, specificity, precision,
                                 fall_out, miss_rate, fdr, npv, auroc, len(tested_sites),
                                 called, float(called) / len(tested_sites),
                                 called_accuracy)


def main():
    parser = argparse.ArgumentParser(description='Calculate call accuracy stats of nn results for cpgs')
    parser.add_argument('--unmethylated', type=str, required=True)
    parser.add_argument('--methylated', type=str, required=True)
    parser.add_argument('--result_file', type=str, required=True,
                        help='a file path to save the evaluation result')
    args = parser.parse_args()

    unmethylated_sites = sample_sites(args.unmethylated, False)
    methylated_sites = sample_sites(args.methylated, True)

    result_file = os.path.abspath(args.result_file)

    pr_writer = open(result_file, 'w')
    pr_writer.write("tested_type\tprob_cf\ttrue_positive\tfalse_positive\ttrue_negative\tfalse_negative\t"
                    "accuracy\trecall\tspecificity\tprecision\t"
                    "fallout\tmiss_rate\tFDR\tNPV\tauc\ttotal_num\tcalled_num\tcalled_ratio\tcalled_accuracy\n")
    for site_num in num_sites:
        tested_sites = methylated_sites[:site_num] + unmethylated_sites[:site_num]
        for prob_cf in prob_cfs:
            pr_writer.write("\t".join(["_" + str(site_num), "%.3f" % prob_cf,
                                       _evaluate_(tested_sites, prob_cf)]) + "\n")
    tested_sites = methylated_sites + unmethylated_sites
    prob_cf = 0.0
    pr_writer.write("\t".join(["all_sites", "%.3f" % prob_cf, _evaluate_(tested_sites, prob_cf)]) + "\n")

    pr_writer.flush()
    pr_writer.close()


if __name__ == '__main__':
    sys.exit(main())
