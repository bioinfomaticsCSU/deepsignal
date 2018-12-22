#! /usr/bin/python
import argparse
import os
import sys
import time

import h5py
import numpy as np
from statsmodels import robust

from _ref_reader import DNAReference
from _utils import get_refloc_of_methysite_in_motif
from _utils import max_num_bases

# analysis_group_path = 'Analyses'
reads_group = 'Raw/Reads'
MAX_BASE_NUM = max_num_bases()

normalize_ways = ['ZScore', 'mad']
normalize_way = 'mad'


def calculate_map_quality():
    pass


def get_alignment_attrs_of_each_strand(strand_path, h5obj):
    strand_basecall_group_alignment = h5obj['/'.join([strand_path, 'Alignment'])]
    alignment_attrs = strand_basecall_group_alignment.attrs
    # attr_names = list(alignment_attrs.keys())

    if strand_path.endswith('template'):
        strand = 't'
    else:
        strand = 'c'
    if sys.version_info[0] >= 3:
        try:
            alignstrand = str(alignment_attrs['mapped_strand'], 'utf-8')
            chrom = str(alignment_attrs['mapped_chrom'], 'utf-8')
        except TypeError:
            alignstrand = str(alignment_attrs['mapped_strand'])
            chrom = str(alignment_attrs['mapped_chrom'])
    else:
        alignstrand = str(alignment_attrs['mapped_strand'])
        chrom = str(alignment_attrs['mapped_chrom'])
    chrom_start = alignment_attrs['mapped_start']
    # print(strand, alignstrand, chrom, type(chrom_start))

    return strand, alignstrand, chrom, chrom_start


def get_readid_from_fast5(h5file):
    first_read = list(h5file[reads_group].keys())[0]
    if sys.version_info[0] >= 3:
        read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'], 'utf-8')
    else:
        read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    # print(read_id)
    return read_id


def get_alignment_info_from_fast5(fast5_path, corrected_group='RawGenomeCorrected_000',
                                  basecall_subgroup='BaseCalled_template',
                                  map_quality=-1):
    try:
        h5file = h5py.File(fast5_path, mode='r')
        corrgroup_path = '/'.join(['Analyses', corrected_group])

        # strand_basecall_group_names = list(h5file[corrgroup_path].keys())
        # for sgname in strand_basecall_group_names:
        #     get_alignment_of_each_strand('/'.join([corrgroup_path, sgname]), h5file)

        if '/'.join([corrgroup_path, basecall_subgroup, 'Alignment']) in h5file:
            fileprefix = os.path.basename(fast5_path).split('.fast5')[0]
            readname = get_readid_from_fast5(h5file)
            strand, alignstrand, chrom, chrom_start = get_alignment_attrs_of_each_strand('/'.join([corrgroup_path,
                                                                                                   basecall_subgroup]),
                                                                                         h5file)

            h5file.close()
            return fileprefix, readname, strand, alignstrand, chrom, chrom_start
        else:
            return '', '', '', '', '', ''
    except IOError:
        print("the {} can't be opened".format(fast5_path))
        return '', '', '', '', '', ''


def get_genomebase2locs_from_label(label_path):
    genomeseq = []
    locs = []
    with open(label_path, 'r') as rf:
        for line in rf:
            words = line.strip().split(' ')
            genomeseq.append(words[2])
            locs.append((int(words[0]), int(words[1])))
    genomeseq = ''.join(genomeseq)
    return genomeseq, locs


def get_signals_from_signal(signal_path):
    # signals = []
    with open(signal_path, 'r') as rf:
        line = next(rf).strip()
        signals = list(map(int, line.split(' ')))
    signals = np.array(signals)
    return signals


def normalize_signals(signals):
    if normalize_way == 'zscore':
        sshift, sscale = np.mean(signals), np.float(np.std(signals))
    elif normalize_way == 'mad':
        sshift, sscale = np.median(signals), np.float(robust.mad(signals))
    else:
        raise ValueError("")
    norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def get_base2signal_from_signal(sl_folder, file_prefix):
    """

    :param sl_folder: folder path of the signal and label files
    :param file_prefix:
    :return:
    """
    signal_path = sl_folder + '/' + file_prefix + '.signal'
    label_path = sl_folder + '/' + file_prefix + '.label'

    if os.path.isfile(signal_path) and os.path.isfile(label_path):
        genomeseq, locs = get_genomebase2locs_from_label(label_path)
        signals = get_signals_from_signal(signal_path)
        signals = normalize_signals(signals)
        signal_list = []
        for locpair in locs:
            signal_list.append(signals[locpair[0]:locpair[1]])
        return genomeseq, signal_list
    return None, None


def convert_motif_seq(ori_seq):
    alphabets = {'A': ['A', ], 'T': ['T', ], 'C': ['C', ], 'G': ['G', ],
                 'N': ['N', ], 'H': ['A', 'C', 'T']}
    outbases = []
    for bbase in ori_seq:
        outbases.append(alphabets[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 2:
            pseqs = []
            for fbase in bases_list[0]:
                for sbase in bases_list[1]:
                    pseqs.append(fbase + sbase)
            return pseqs
        else:
            pseqs = recursive_permute(bases_list[1:])
            pseq_list = [bases_list[0], pseqs]
            return recursive_permute(pseq_list)
    return recursive_permute(outbases)


def extract_cpgsites_signal_features(fast5_path, sl_folder,
                                     chrom2len,
                                     methy_label='-1',
                                     num_bases=10,
                                     corrected_group='RawGenomeCorrected_000',
                                     basecall_subgroup='BaseCalled_template',
                                     motifs='CG',
                                     methyloc=0,
                                     map_quality=-1):
    """
    id	label	readname    contig	read_strand	align_strand    pos pos_in_strand	k-mer(17-mer) signal
    :param fast5_path:
    :param sl_folder:
    :param chrom2len:
    :param methy_label:
    :param num_bases:
    :param corrected_group:
    :param basecall_subgroup:
    :param motifs:
    :param methyloc:
    :param map_quality:
    :return:
    """
    fileprefix, readname, strand, alignstrand, chrom, chrom_start = get_alignment_info_from_fast5(fast5_path,
                                                                                                  corrected_group,
                                                                                                  basecall_subgroup,
                                                                                                  map_quality)
    cpgsite_info = None
    if fileprefix != '':
        genomeseq, signal_list = get_base2signal_from_signal(sl_folder, fileprefix)

        if genomeseq is not None:
            chromlen = chrom2len[chrom]
            if alignstrand == '+':
                chrom_start_in_alignstrand = chrom_start
            else:
                chrom_start_in_alignstrand = chromlen - (chrom_start + len(genomeseq))

            cpgsite_info = []

            # cpg_site_locs = get_motif_start_sites(genomeseq, 'CG')
            ori_motif_seqs = motifs.split(',')

            motif_seqs = []
            for ori_motif in ori_motif_seqs:
                motif_seqs += convert_motif_seq(ori_motif)

            cpg_site_locs = []
            for mseq in motif_seqs:
                cpg_site_locs += get_refloc_of_methysite_in_motif(genomeseq, mseq, methyloc)

            for cpgloc_in_read in cpg_site_locs:
                if num_bases <= cpgloc_in_read < len(genomeseq) - num_bases:
                    cpgloc_in_ref = cpgloc_in_read + chrom_start_in_alignstrand
                    # cpgid = readname+strand+str(cpgloc_in_ref)
                    cpgid = readname + chrom + alignstrand + str(cpgloc_in_ref) + strand
                    if alignstrand == '-':
                        pos = chromlen - 1 - cpgloc_in_ref
                    else:
                        pos = cpgloc_in_ref

                    k_mer = genomeseq[cpgloc_in_read-num_bases:cpgloc_in_read+num_bases+1]
                    k_signals = ';'.join([','.join(list(map(str, x))) for x in
                                          signal_list[cpgloc_in_read-num_bases:cpgloc_in_read+num_bases+1]])
                    cpgsite_info.append('\t'.join([cpgid, methy_label, readname, chrom, strand,
                                                   alignstrand, str(pos), str(cpgloc_in_ref), k_mer, k_signals]))

    return cpgsite_info


def extract_cpg_candidates_signal_features():
    parser = argparse.ArgumentParser(description='extract signal features of cpg candidates from fast5 and '
                                                 '.label/.signal files')
    parser.add_argument('--fast5_dir', type=str, required=True)
    parser.add_argument('--sl_dir', type=str, required=True,
                        help='path where the .label/.signal files are. Generated by chiron')
    parser.add_argument('--reference_path', type=str, required=True)
    parser.add_argument('--write_path', type=str, required=False,
                        default='signal_features.tsv',
                        help='file path to save the results')

    parser.add_argument('--methyed_label', type=str,
                        required=False, default='-1', help="'0' or '1'")
    parser.add_argument('--num_bases', type=int, required=False, default=10,
                        help='num of bases in one side of the cpg candidates. default 10')
    parser.add_argument('--corrected_group', type=str, required=False,
                        default='RawGenomeCorrected_000',
                        help='default RawGenomeCorrected_000')
    parser.add_argument('--basecall_subgroup', type=str, required=False,
                        default='BaseCalled_template',
                        help='default BaseCalled_template; BaseCalled_complement')
    parser.add_argument('--map_quality', type=int, required=False, default=-1,
                        help='(not used) threshold for the map quality of the alignment. a alignment whose '
                             'map quality is smaller than this value is abandoned. default -1')
    parser.add_argument('--motifs', type=str, required=False,
                        default='CG',
                        help='motif seq, default: CG. can be multi motifs splited by comma, but the methy_loc must be'
                             ' the same')
    parser.add_argument('--methy_loc_in_motif', type=int, required=False,
                        default=0,
                        help='0-based location of the methylation base in the motif, default 0')
    args = parser.parse_args()

    fast5_folder = args.fast5_dir
    sl_folder = args.sl_dir
    ref_path = args.reference_path
    methy_label = args.methyed_label
    w_filepath = args.write_path
    num_bases_cutoff = args.num_bases
    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    motif_seqs = args.motifs
    methyloc = args.methy_loc_in_motif
    mapq = args.map_quality

    start = time.time()
    refseq = DNAReference(ref_path)
    chrom2len = {}
    for contigname in refseq.getcontignames():
        chrom2len[contigname] = len(refseq.getcontigs()[contigname])
    del refseq

    wf = open(w_filepath, 'w')
    wf.write('\t'.join(['id', 'label', 'readname', 'contig', 'read_strand', 'align_strand',
                        'pos', 'pos_in_strand', 'k-mer', 'k-signal']) + '\n')
    count = 1
    for fast5_sub in os.listdir(fast5_folder):
        sub_path = '/'.join([fast5_folder, fast5_sub])
        if os.path.isdir(sub_path):
            sl_sub_path = '/'.join([sl_folder, fast5_sub])
            for sub2_path in os.listdir(sub_path):
                f5file = '/'.join([sub_path, sub2_path])
                cpgsites_info = extract_cpgsites_signal_features(f5file, sl_sub_path, chrom2len,
                                                                 methy_label, num_bases_cutoff,
                                                                 corrected_group,
                                                                 basecall_subgroup,
                                                                 motif_seqs,
                                                                 methyloc)
                if cpgsites_info is not None:
                    for cpgsite_info in cpgsites_info:
                        wf.write(str(cpgsite_info) + '\n')
                print('{}th read done..'.format(count))
                count += 1
        else:
            cpgsites_info = extract_cpgsites_signal_features(sub_path, sl_folder, chrom2len,
                                                             methy_label, num_bases_cutoff,
                                                             corrected_group,
                                                             basecall_subgroup,
                                                             motif_seqs,
                                                             methyloc)
            if cpgsites_info is not None:
                for cpgsite_info in cpgsites_info:
                    wf.write(str(cpgsite_info) + '\n')
            print('{}th read done..'.format(count))
            count += 1
    wf.flush()
    wf.close()
    print('cost time: {} seconds..'.format(time.time()-start))


def main():
    extract_cpg_candidates_signal_features()
    pass


if __name__ == '__main__':
    main()
    pass
