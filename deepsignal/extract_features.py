"""the feature extraction module of deepsignal.
output format:
chrom, pos, alignstrand, pos_in_strand, readname, read_strand, k_mer, signal_means,
signal_stds, signal_lens, cent_signals, methy_label
"""

from __future__ import absolute_import

import sys
import os
import argparse
import time
import h5py
import random
import numpy as np
import multiprocessing as mp
from .utils.process_utils import Queue
from statsmodels import robust

from .utils.process_utils import str2bool
from .utils.process_utils import get_fast5s
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import get_motif_seqs

from .utils.ref_reader import get_contig2len

reads_group = 'Raw/Reads'
queen_size_border = 2000
time_wait = 3
# MAX_LEGAL_SIGNAL_NUM = 800  # 800 only for 17-mer

key_sep = "||"


def _get_label_raw(fast5_fn, correct_group, correct_subgroup):
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        raw_dat = list(fast5_data[reads_group].values())[0]
        # raw_attrs = raw_dat.attrs
        # raw_dat = raw_dat['Signal'].value
        raw_dat = raw_dat['Signal'][()]
    except Exception:
        raise RuntimeError('Raw data is not stored in Raw/Reads/Read_[read#] so '
                           'new segments cannot be identified.')

    # Get Events
    try:
        event = fast5_data['/Analyses/' + correct_group + '/' + correct_subgroup + '/Events']
        # print(event)
    except Exception:
        raise RuntimeError('events not found.')

    try:
        corr_attrs = dict(list(event.attrs.items()))
        read_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']
        # print('read_start_rel_to_raw: ',read_start_rel_to_raw)
        starts = list(map(lambda x: x + read_start_rel_to_raw, event['start']))
    except KeyError:
        # starts = list(map(lambda x: x, event['start']))
        raise KeyError('no read_start_rel_to_raw in event attributes')

    lengths = event['length'].astype(np.int)
    base = [x.decode("UTF-8") for x in event['base']]
    assert len(starts) == len(lengths)
    assert len(lengths) == len(base)
    events = list(zip(starts, lengths, base))
    return raw_dat, events


def _get_alignment_attrs_of_each_strand(strand_path, h5obj):
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
            alignstrand = alignment_attrs['mapped_strand']
            alignstrand = str(alignstrand) if type(alignstrand) is not str else alignstrand
            chrom = alignment_attrs['mapped_chrom']
            chrom_val_type = type(chrom)
            chrom = str(chrom) if chrom_val_type is not str else chrom
            if chrom_val_type is bytes and chrom.startswith("b'"):
                chrom = chrom.split("'")[1]
    else:
        alignstrand = str(alignment_attrs['mapped_strand'])
        chrom = alignment_attrs['mapped_chrom']
        chrom_val_type = type(chrom)
        chrom = str(chrom) if chrom_val_type is not str else chrom
        if chrom_val_type is bytes and chrom.startswith("b'"):
            chrom = chrom.split("'")[1]
    chrom_start = alignment_attrs['mapped_start']

    return strand, alignstrand, chrom, chrom_start


def _get_readid_from_fast5(h5file):
    first_read = list(h5file[reads_group].keys())[0]
    if sys.version_info[0] >= 3:
        try:
            read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'], 'utf-8')
        except TypeError:
            read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    else:
        read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    # print(read_id)
    return read_id


def _get_alignment_info_from_fast5(fast5_path, corrected_group='RawGenomeCorrected_000',
                                   basecall_subgroup='BaseCalled_template'):
    try:
        h5file = h5py.File(fast5_path, mode='r')
        corrgroup_path = '/'.join(['Analyses', corrected_group])

        if '/'.join([corrgroup_path, basecall_subgroup, 'Alignment']) in h5file:
            # fileprefix = os.path.basename(fast5_path).split('.fast5')[0]
            readname = _get_readid_from_fast5(h5file)
            strand, alignstrand, chrom, chrom_start = _get_alignment_attrs_of_each_strand('/'.join([corrgroup_path,
                                                                                                    basecall_subgroup]),
                                                                                          h5file)

            h5file.close()
            return readname, strand, alignstrand, chrom, chrom_start
        else:
            return '', '', '', '', ''
    except IOError:
        print("the {} can't be opened".format(fast5_path))
        return '', '', '', '', ''


def _normalize_signals(signals, normalize_method="mad"):
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), np.float(np.std(signals))
    elif normalize_method == 'mad':
        sshift, sscale = np.median(signals), np.float(robust.mad(signals))
    else:
        raise ValueError("")
    norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def _get_central_signals(signals_list, rawsignal_num=360):
    signal_lens = [len(x) for x in signals_list]

    if sum(signal_lens) < rawsignal_num:
        # real_signals = sum(signals_list, [])
        real_signals = np.concatenate(signals_list)
        cent_signals = np.append(real_signals, np.array([0] * (rawsignal_num - len(real_signals))))
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

            # left_signals = sum(signals_list[:mid_loc], [])
            # right_signals = sum(signals_list[mid_loc:], [])
            left_signals = np.concatenate(signals_list[:mid_loc])
            right_signals = np.concatenate(signals_list[mid_loc:])

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
                cent_signals = np.append(left_signals[-left_len:], right_signals[:right_len])
    return cent_signals


def _get_scaling_of_a_read(fast5fp):
    global_key = "UniqueGlobalKey/"
    try:
        h5file = h5py.File(fast5fp, mode='r')
        channel_info = dict(list(h5file[global_key + 'channel_id'].attrs.items()))
        digi = channel_info['digitisation']
        parange = channel_info['range']
        offset = channel_info['offset']
        scaling = parange / digi

        h5file.close()
        # print(scaling, offset)
        return scaling, offset
    except IOError:
        print("the {} can't be opened".format(fast5fp))
        return None, None


def _rescale_signals(rawsignals, scaling, offset):
    return np.array(scaling * (rawsignals + offset), dtype=np.float)


def _extract_features(fast5s, corrected_group, basecall_subgroup, normalize_method,
                      motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len,
                      methy_label, positions):
    features_list = []
    error = 0
    for fast5_fp in fast5s:
        try:
            raw_signal, events = _get_label_raw(fast5_fp, corrected_group, basecall_subgroup)

            scaling, offset = _get_scaling_of_a_read(fast5_fp)
            if scaling is None:
                continue
            else:
                raw_signal = _rescale_signals(raw_signal, scaling, offset)

            norm_signals = _normalize_signals(raw_signal, normalize_method)
            genomeseq, signal_list = "", []
            for e in events:
                genomeseq += str(e[2])
                signal_list.append(norm_signals[e[0]:(e[0] + e[1])])

            readname, strand, alignstrand, chrom, \
                chrom_start = _get_alignment_info_from_fast5(fast5_fp, corrected_group, basecall_subgroup)

            chromlen = chrom2len[chrom]
            if alignstrand == '+':
                chrom_start_in_alignstrand = chrom_start
            else:
                chrom_start_in_alignstrand = chromlen - (chrom_start + len(genomeseq))

            # tsite_locs = []
            # for mseq in motif_seqs:
            #     tsite_locs += get_refloc_of_methysite_in_motif(genomeseq, mseq, methyloc)
            tsite_locs = get_refloc_of_methysite_in_motif(genomeseq, set(motif_seqs), methyloc)

            if kmer_len % 2 == 0:
                raise ValueError("kmer_len must be odd")
            num_bases = (kmer_len - 1) // 2

            for loc_in_read in tsite_locs:
                if num_bases <= loc_in_read < len(genomeseq) - num_bases:
                    loc_in_ref = loc_in_read + chrom_start_in_alignstrand

                    # cpgid = readname + chrom + alignstrand + str(cpgloc_in_ref) + strand
                    if alignstrand == '-':
                        pos = chromlen - 1 - loc_in_ref
                    else:
                        pos = loc_in_ref

                    if (positions is not None) and (key_sep.join([chrom, str(pos), alignstrand]) not in positions):
                        continue

                    k_mer = genomeseq[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    k_signals = signal_list[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]

                    signal_lens = [len(x) for x in k_signals]
                    # if sum(signal_lens) > MAX_LEGAL_SIGNAL_NUM:
                    #     continue

                    signal_means = [np.mean(x) for x in k_signals]
                    signal_stds = [np.std(x) for x in k_signals]

                    cent_signals = _get_central_signals(k_signals, raw_signals_len)

                    features_list.append((chrom, pos, alignstrand, loc_in_ref, readname, strand,
                                          k_mer, signal_means, signal_stds, signal_lens,
                                          cent_signals, methy_label))
        except Exception:
            error += 1
            continue
    # print("extracted success {} of {}".format(len(fast5s) - error, len(fast5s)))
    # print("features_str len {}".format(len(features_str)))
    return features_list, error


def _features_to_str(features):
    """

    :param features: a tuple
    :return:
    """
    chrom, pos, alignstrand, loc_in_ref, readname, strand, k_mer, signal_means, signal_stds, \
        signal_lens, cent_signals, methy_label = features
    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    signal_len_text = ','.join([str(x) for x in signal_lens])
    cent_signals_text = ','.join([str(x) for x in cent_signals])

    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, strand, k_mer, means_text,
                      stds_text, signal_len_text, cent_signals_text, str(methy_label)])


def _fill_files_queue(fast5s_q, fast5_files, batch_size):
    for i in np.arange(0, len(fast5_files), batch_size):
        fast5s_q.put(fast5_files[i:(i+batch_size)])
    return


def get_a_batch_features_str(fast5s_q, featurestr_q, errornum_q,
                             corrected_group, basecall_subgroup, normalize_method,
                             motif_seqs, methyloc, chrom2len, kmer_len, raw_signals_len, methy_label,
                             positions):
    while True:
        fast5s = fast5s_q.get()
        if fast5s == "kill":
            fast5s_q.put("kill")
            break
        features_list, error_num = _extract_features(fast5s, corrected_group, basecall_subgroup,
                                                     normalize_method, motif_seqs, methyloc,
                                                     chrom2len, kmer_len, raw_signals_len, methy_label,
                                                     positions)
        features_str = []
        for features in features_list:
            features_str.append(_features_to_str(features))

        errornum_q.put(error_num)
        featurestr_q.put(features_str)
        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)
    # print("get out of while loop in get_feature_str")


def _write_featurestr_to_file(write_fp, featurestr_q):
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep(time_wait)
            if featurestr_q.empty():
                time.sleep(time_wait)
                continue
            features_str = featurestr_q.get()
            if features_str == "kill":
                break
            for one_features_str in features_str:
                wf.write(one_features_str + "\n")
            wf.flush()


def _write_featurestr_to_dir(write_dir, featurestr_q, w_batch_num):
    if os.path.exists(write_dir):
        if os.path.isfile(write_dir):
            raise FileExistsError("{} already exists as a file, please use another write_dir".format(write_dir))
    else:
        os.makedirs(write_dir)

    file_count = 0
    wf = open("/".join([write_dir, str(file_count) + ".tsv"]), "w")
    batch_count = 0
    while True:
        # during test, it's ok without the sleep(time_wait)
        if featurestr_q.empty():
            time.sleep(time_wait)
            continue
        features_str = featurestr_q.get()
        if features_str == "kill":
            break

        if batch_count >= w_batch_num:
            wf.flush()
            wf.close()
            file_count += 1
            wf = open("/".join([write_dir, str(file_count) + ".tsv"]), "w")
            batch_count = 0
        for one_features_str in features_str:
            wf.write(one_features_str + "\n")
        batch_count += 1


def _write_featurestr(write_fp, featurestr_q, w_batch_num=10000, is_dir=False):
    if is_dir:
        _write_featurestr_to_dir(write_fp, featurestr_q, w_batch_num)
    else:
        _write_featurestr_to_file(write_fp, featurestr_q)


def _read_position_file(position_file):
    postions = set()
    with open(position_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            postions.add(key_sep.join(words[:3]))
    return postions


def _extract_preprocess(fast5_dir, is_recursive, motifs, is_dna, reference_path, f5_batch_num,
                        position_file):

    fast5_files = get_fast5s(fast5_dir, is_recursive)
    print("{} fast5 files in total..".format(len(fast5_files)))

    print("parse the motifs string..")
    motif_seqs = get_motif_seqs(motifs, is_dna)

    print("read genome reference file..")
    chrom2len = get_contig2len(reference_path)

    print("read position file if it is not None..")
    positions = None
    if position_file is not None:
        positions = _read_position_file(position_file)

    # fast5s_q = mp.Queue()
    fast5s_q = Queue()
    _fill_files_queue(fast5s_q, fast5_files, f5_batch_num)

    return motif_seqs, chrom2len, fast5s_q, len(fast5_files), positions


def extract_features(fast5_dir, is_recursive, reference_path, is_dna,
                     batch_size, write_fp, nproc,
                     corrected_group, basecall_subgroup, normalize_method,
                     motifs, methyloc, kmer_len, raw_signals_len, methy_label,
                     position_file, w_is_dir, w_batch_num):
    start = time.time()

    motif_seqs, chrom2len, fast5s_q, len_fast5s, positions = _extract_preprocess(fast5_dir, is_recursive,
                                                                                 motifs, is_dna, reference_path,
                                                                                 batch_size, position_file)

    # featurestr_q = mp.Queue()
    # errornum_q = mp.Queue()
    featurestr_q = Queue()
    errornum_q = Queue()

    featurestr_procs = []
    if nproc > 1:
        nproc -= 1
    fast5s_q.put("kill")
    for _ in range(nproc):
        p = mp.Process(target=get_a_batch_features_str, args=(fast5s_q, featurestr_q, errornum_q,
                                                              corrected_group, basecall_subgroup,
                                                              normalize_method, motif_seqs,
                                                              methyloc, chrom2len, kmer_len, raw_signals_len,
                                                              methy_label, positions))
        p.daemon = True
        p.start()
        featurestr_procs.append(p)

    print("write_process started..")
    p_w = mp.Process(target=_write_featurestr, args=(write_fp, featurestr_q, w_batch_num, w_is_dir))
    p_w.daemon = True
    p_w.start()

    errornum_sum = 0
    while True:
        # print("killing feature_p")
        running = any(p.is_alive() for p in featurestr_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in featurestr_procs:
        p.join()

    print("finishing the write_process..")
    featurestr_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed..\n"
          "extract_features costs %.1f seconds.." % (errornum_sum, len_fast5s,
                                                     time.time() - start))


def main():
    extraction_parser = argparse.ArgumentParser("extract features from corrected (tombo) fast5s for "
                                                "training or testing."
                                                "\nIt is suggested that running this module 1 flowcell a time, "
                                                "or a group of flowcells a time, "
                                                "if the whole data is extremely large.")
    ep_input = extraction_parser.add_argument_group("INPUT")
    ep_input.add_argument("--fast5_dir", "-i", action="store", type=str,
                          required=True,
                          help="the directory of fast5 files")
    ep_input.add_argument("--recursively", "-r", action="store", type=str, required=False,
                          default='yes',
                          help='is to find fast5 files from fast5_dir recursively. '
                               'default true, t, yes, 1')
    ep_input.add_argument("--corrected_group", action="store", type=str, required=False,
                          default='RawGenomeCorrected_000',
                          help='the corrected_group of fast5 files after '
                               'tombo re-squiggle. default RawGenomeCorrected_000')
    ep_input.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                          default='BaseCalled_template',
                          help='the corrected subgroup of fast5 files. default BaseCalled_template')
    ep_input.add_argument("--reference_path", action="store",
                          type=str, required=True,
                          help="the reference file to be used, usually is a .fa file")
    ep_input.add_argument("--is_dna", action="store", type=str, required=False,
                          default='yes',
                          help='whether the fast5 files from DNA sample or not. '
                               'default true, t, yes, 1. '
                               'set this option to no/false/0 if '
                               'the fast5 files are from RNA sample.')

    ep_extraction = extraction_parser.add_argument_group("EXTRACTION")
    ep_extraction.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                               default="mad", required=False,
                               help="the way for normalizing signals in read level. "
                                    "mad or zscore, default mad")
    ep_extraction.add_argument("--methy_label", action="store", type=int,
                               choices=[1, 0], required=False, default=1,
                               help="the label of the interested modified bases, this is for training."
                                    " 0 or 1, default 1")
    ep_extraction.add_argument("--kmer_len", action="store",
                               type=int, required=False, default=17,
                               help="len of kmer. default 17")
    ep_extraction.add_argument("--cent_signals_len", action="store",
                               type=int, required=False, default=360,
                               help="the number of signals to be used in deepsignal, default 360")
    ep_extraction.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    ep_extraction.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    # ep_extraction.add_argument("--region", action="store", type=str,
    #                            required=False, default=None,
    #                            help="region of interest, e.g.: chr1:0-10000, default None, "
    #                                 "for the whole region")
    ep_extraction.add_argument("--positions", action="store", type=str,
                               required=False, default=None,
                               help="file with a list of positions interested (must be formatted as tab-separated file"
                                    " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                                    "need to be set. --positions is used to narrow down the range of the trageted "
                                    "motif locs. default None")

    ep_output = extraction_parser.add_argument_group("OUTPUT")
    ep_output.add_argument("--write_path", "-o", action="store",
                           type=str, required=True,
                           help='file path to save the features')
    ep_output.add_argument("--w_is_dir", action="store",
                           type=str, required=False, default="no",
                           help='if using a dir to save features into multiple files')
    ep_output.add_argument("--w_batch_num", action="store",
                           type=int, required=False, default=200,
                           help='features batch num to save in a single writed file when --is_dir is true')

    extraction_parser.add_argument("--nproc", "-p", action="store", type=int, default=1,
                                   required=False,
                                   help="number of processes to be used, default 1")
    extraction_parser.add_argument("--f5_batch_num", action="store", type=int, default=100,
                                   required=False,
                                   help="number of files to be processed by each process one time, default 100")

    extraction_args = extraction_parser.parse_args()

    fast5_dir = extraction_args.fast5_dir
    is_recursive = str2bool(extraction_args.recursively)

    corrected_group = extraction_args.corrected_group
    basecall_subgroup = extraction_args.basecall_subgroup
    normalize_method = extraction_args.normalize_method

    reference_path = extraction_args.reference_path
    is_dna = str2bool(extraction_args.is_dna)
    write_path = extraction_args.write_path
    w_is_dir = str2bool(extraction_args.w_is_dir)
    w_batch_num = extraction_args.w_batch_num

    kmer_len = extraction_args.kmer_len
    cent_signals_num = extraction_args.cent_signals_len
    motifs = extraction_args.motifs
    mod_loc = extraction_args.mod_loc
    methy_label = extraction_args.methy_label
    position_file = extraction_args.positions

    nproc = extraction_args.nproc
    f5_batch_num = extraction_args.f5_batch_num

    extract_features(fast5_dir, is_recursive, reference_path, is_dna,
                     f5_batch_num, write_path, nproc, corrected_group, basecall_subgroup,
                     normalize_method, motifs, mod_loc, kmer_len, cent_signals_num, methy_label,
                     position_file, w_is_dir, w_batch_num)


if __name__ == '__main__':
    sys.exit(main())
