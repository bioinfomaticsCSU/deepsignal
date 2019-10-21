from __future__ import absolute_import
import fnmatch
import os
import multiprocessing
import multiprocessing.queues

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
