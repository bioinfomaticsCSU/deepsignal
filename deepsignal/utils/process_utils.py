from __future__ import absolute_import
import fnmatch
import os
import random
import multiprocessing
import multiprocessing.queues
import numpy as np
import gc
import struct
import math

basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
                 'Z': 'Z'}

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
code2base_dna = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
base2code_rna = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}
code2base_rna = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'N'}

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}
iupac_alphabets_rna = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'U': ['U'],
                       'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                       'Y': ['C', 'U'], 'K': ['G', 'U'], 'W': ['A', 'U'],
                       'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
                       'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
                       'N': ['A', 'C', 'G', 'U']}

# max_queue_size = 2000


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def _alphabet(letter, dbasepairs):
    if letter in dbasepairs.keys():
        return dbasepairs[letter]
    return 'N'


def complement_seq(base_seq, seq_type="DNA"):
    rbase_seq = base_seq[::-1]
    comseq = ''
    try:
        if seq_type == "DNA":
            comseq = ''.join([_alphabet(x, basepairs) for x in rbase_seq])
        elif seq_type == "RNA":
            comseq = ''.join([_alphabet(x, basepairs_rna) for x in rbase_seq])
        else:
            raise ValueError("the seq_type must be DNA or RNA")
    except Exception:
        print('something wrong in the dna/rna sequence.')
    return comseq


# def get_refloc_of_methysite_in_motif(seqstr, motif='CG', methyloc_in_motif=0):
#     """
#
#     :param seqstr:
#     :param motif:
#     :param methyloc_in_motif: 0-based
#     :return:
#     """
#     strlen = len(seqstr)
#     motiflen = len(motif)
#     sites = []
#     for i in range(0, strlen - motiflen + 1):
#         if seqstr[i:i + motiflen] == motif:
#             sites.append(i+methyloc_in_motif)
#     return sites


def get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif=0):
    """

    :param seqstr:
    :param motifset:
    :param methyloc_in_motif: 0-based
    :return:
    """
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] in motifset:
            sites.append(i+methyloc_in_motif)
    return sites


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            outbases.append(iupac_alphabets_rna[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        elif len(bases_list) == 2:
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


def get_motif_seqs(motifs, is_dna=True):
    ori_motif_seqs = motifs.strip().split(',')

    motif_seqs = []
    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif.strip().upper(), is_dna)
    return motif_seqs


def get_fast5s(fast5_dir, is_recursive=True):
    fast5_dir = os.path.abspath(fast5_dir)
    fast5s = []
    if is_recursive:
        for root, dirnames, filenames in os.walk(fast5_dir):
            for filename in fnmatch.filter(filenames, '*.fast5'):
                fast5_path = os.path.join(root, filename)
                fast5s.append(fast5_path)
    else:
        for fast5_name in os.listdir(fast5_dir):
            if fast5_name.endswith('.fast5'):
                fast5_path = '/'.join([fast5_dir, fast5_name])
                fast5s.append(fast5_path)
    return fast5s


def count_line_num(sl_filepath, fheader=False):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    print('done count the lines of file {}'.format(sl_filepath))
    return count


def random_select_file_rows(ori_file, w_file, w_other_file=None, maxrownum=100000000, header=False):
    """

    :param ori_file:
    :param w_file:
    :param w_other_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for _ in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    if w_other_file is not None:
        wlf = open(w_other_file, 'w')
    with open(ori_file) as rf:
        if header:
            lineheader = next(rf)
            wf.write(lineheader)
            if w_other_file is not None:
                wlf.write(lineheader)
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1] - 1):
                other_line = next(rf)
                if w_other_file is not None:
                    wlf.write(other_line)
            chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    if w_other_file is not None:
        wlf.close()
    print('random_select_file_rows finished..')


def random_select_file_rows_s(ori_file, w_file, w_other_file, maxrownum=100000000, header=False):
    """
    split line indexs to two arrays randomly, write the two group of lines into two files,
     and return the arrays
    :param ori_file:
    :param w_file:
    :param w_other_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for _ in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    wlf = open(w_other_file, 'w')
    lidxs1, lidxs2 = [], []
    lidx_cnt = 0
    with open(ori_file) as rf:
        if header:
            lineheader = next(rf)
            wf.write(lineheader)
            wlf.write(lineheader)
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1] - 1):
                wlf.write(next(rf))
                lidxs2.append(lidx_cnt)
                lidx_cnt += 1
            chosen_line = next(rf)
            wf.write(chosen_line)
            lidxs1.append(lidx_cnt)
            lidx_cnt += 1
    wf.close()
    wlf.close()
    print('random_select_file_rows_s finished..')
    return lidxs1, lidxs2


def read_one_shuffle_info(filepath, shuffle_lines_num, total_lines_num, checked_lines_num, isheader):
    with open(filepath, 'r') as rf:
        if isheader:
            next(rf)
        count = 0
        while count < checked_lines_num:
            next(rf)
            count += 1

        count = 0
        lines_info = []
        lines_num = min(shuffle_lines_num, (total_lines_num - checked_lines_num))
        for line in rf:
            if count < lines_num:
                lines_info.append(line.strip())
                count += 1
            else:
                break
        print('done reading file {}'.format(filepath))
        return lines_info


def shuffle_samples(samples_info):
    mark = list(range(len(samples_info)))
    np.random.shuffle(mark)
    shuffled_samples = []
    for i in mark:
        shuffled_samples.append(samples_info[i])
    return shuffled_samples


def write_to_one_file_append(features_info, wfilepath):
    with open(wfilepath, 'a') as wf:
        for i in range(0, len(features_info)):
            wf.write(features_info[i] + '\n')
    print('done writing features info to {}'.format(wfilepath))


def concat_two_files(file1, file2, concated_fp, shuffle_lines_num=2000000,
                     lines_num=1000000000000, isheader=False):
    open(concated_fp, 'w').close()

    if isheader:
        rf1 = open(file1, 'r')
        wf = open(concated_fp, 'a')
        wf.write(next(rf1))
        wf.close()
        rf1.close()

    f1line_count = count_line_num(file1, isheader)
    f2line_count = count_line_num(file2, False)

    line_ratio = float(f2line_count) / f1line_count
    shuffle_lines_num2 = round(line_ratio * shuffle_lines_num) + 1

    checked_lines_num1, checked_lines_num2 = 0, 0
    while checked_lines_num1 < lines_num or checked_lines_num2 < lines_num:
        file1_info = read_one_shuffle_info(file1, shuffle_lines_num, lines_num, checked_lines_num1, isheader)
        checked_lines_num1 += len(file1_info)
        file2_info = read_one_shuffle_info(file2, shuffle_lines_num2, lines_num, checked_lines_num2, False)
        checked_lines_num2 += len(file2_info)
        if len(file1_info) == 0 and len(file2_info) == 0:
            break
        samples_info = shuffle_samples(file1_info + file2_info)
        write_to_one_file_append(samples_info, concated_fp)

        del file1_info
        del file2_info
        del samples_info
        gc.collect()
    print('done concating files to: {}'.format(concated_fp))


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
        base_int = [base2code_dna[v] for v in base_char]
        means = [float(v) for v in means.split(',')]
        stds = [float(v) for v in stds.split(',')]
        siglen = [int(v) for v in siglen.split(',')]
        signals = [float(v) for v in signals.split(',')]
        label = int(label)
        bin_writer.write(struct.pack(fmt, *base_int + means + stds + siglen + signals + [label]))
    fread.close()
    bin_writer.close()


# for balancing kmer distri in training samples ===
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


# for balancing kmer distri in training samples ===
def _get_kmer2ratio_n_totalline(kmer_count):
    total_cnt = sum(list(kmer_count.values()))
    kmer_ratios = dict()
    for kmer in kmer_count.keys():
        kmer_ratios[kmer] = float(kmer_count[kmer])/total_cnt
    return kmer_ratios, total_cnt


# for balancing kmer distri in training samples ===
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


# for balancing kmer distri in training samples ===
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


# for balancing kmer distri in training samples ===
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
    print('_write_randsel_lines finished..')


# balance kmer distri in neg_training file as pos_training file
def select_negsamples_asposkmer(pos_file, totalneg_file, seled_neg_file):
    kmer_count = _count_kmers_of_feafile(pos_file)
    kmer2ratio, totalline = _get_kmer2ratio_n_totalline(kmer_count)

    print("{} kmers from kmer2ratio file:{}".format(len(kmer2ratio), pos_file))
    kmer2lines = _get_kmer2lines(totalneg_file)
    sel_lines = _rand_select_by_kmer_ratio(kmer2lines, kmer2ratio, totalline)
    _write_randsel_lines(totalneg_file, seled_neg_file, sel_lines)


class SharedCounter(object):
    """ A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


class Queue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, ctx=multiprocessing.get_context(), **kwargs)
        self._size = SharedCounter(0)

    def put(self, *args, **kwargs):
        super(Queue, self).put(*args, **kwargs)
        self._size.increment(1)

    def get(self, *args, **kwargs):
        self._size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self) -> int:
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self._size.value

    def empty(self) -> bool:
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return self.qsize() == 0
