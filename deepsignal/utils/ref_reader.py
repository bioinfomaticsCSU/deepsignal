from __future__ import absolute_import

from deepsignal.utils.process_utils import complement_seq
from deepsignal.utils.process_utils import get_refloc_of_methysite_in_motif


def get_contig2len(ref_path):
    refseq = DNAReference(ref_path)
    chrom2len = {}
    for contigname in refseq.getcontignames():
        chrom2len[contigname] = len(refseq.getcontigs()[contigname])
    del refseq
    return chrom2len


def get_contigs_of_ref(reffile):
    contig2seq = {}
    with open(reffile, 'r') as rf:
        contigname = ''
        contigseq = ''
        for line in rf:
            if line.startswith('>'):
                if contigname != '' and contigseq != '':
                    contig2seq[contigname] = contigseq
                contigname = line.strip()[1:].split(' ')[0]
                contigseq = ''
            else:
                contigseq += line.strip()
        contig2seq[contigname] = contigseq
    return contig2seq


class DNAReference:
    def __init__(self, reffile):
        self._contignames = []
        self._contigs = {}  # contigname 2 contigseq
        with open(reffile, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        self._contigs[contigname] = contigseq
                        self._contignames.append(contigname)
                    contigname = line.strip()[1:].split(' ')[0]
                    contigseq = ''
                else:
                    # turn to upper case
                    contigseq += line.strip().upper()
            self._contigs[contigname] = contigseq
            self._contignames.append(contigname)

    def getcontigs(self):
        return self._contigs

    def getcontignames(self):
        return self._contignames


class DNAContig:
    def __init__(self, contigname, contigseq):
        self._name = contigname
        self._seq = contigseq
        self._len = len(contigseq)
        self._complementseq = complement_seq(contigseq)

    def getseq(self):
        return self._seq

    def getlen(self):
        return self._len

    def getcomplementseq(self):
        return self._complementseq

    def getname(self):
        return self._name

    def get_seq_CpG_sites(self):
        return get_refloc_of_methysite_in_motif(self._seq, 'CG', 0)

    def get_comseq_CpG_sites(self):
        return get_refloc_of_methysite_in_motif(self._complementseq, 'CG', 0)

    def get_subseq_start_sites_of_seq(self, subseq, offsetloc=0):
        return get_refloc_of_methysite_in_motif(self._seq, subseq, offsetloc)

    def get_subseq_start_sites_of_comseq(self, subseq, offsetloc=0):
        return get_refloc_of_methysite_in_motif(self._complementseq, subseq, offsetloc)
