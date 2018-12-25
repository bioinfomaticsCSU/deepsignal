#! /usr/bin/python
from _utils import complement_seq
from _utils import get_refloc_of_methysite_in_motif


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


# --------------get all cpg candidates from reference----------------------------
def get_reference_cpgsites(reffile):
    refcontig2seq = get_contigs_of_ref(reffile)
    contig2cpginfo = {}

    for contig in refcontig2seq.keys():
        contig2cpginfo[contig] = (get_contig_cpgsites(refcontig2seq[contig]),
                                  get_contig_cpgsites(complement_seq(refcontig2seq[contig])))
    return contig2cpginfo


def get_contig_cpgsites(contig_str):
    """

    :param contig_str:
    :return: key: cpg site position (0-based leftmost position in reference),
    value: (11-(at most)-mer around the cpg site, positions of each 6mer)
    """
    cpgsite2cpginfo = {}
    contigstrlen = len(contig_str)
    for i in range(0, contigstrlen-1):
        if contig_str[i:i+2] == 'CG':
            mer_11_s, mer_11_e = max(0, i-5), min(contigstrlen, i+5)
            mer_6_s, mer_6_e = max(0, i-5), min(contigstrlen, i + 5) - 5
            cpgsite2cpginfo[i] = (contig_str[mer_11_s:mer_11_e+1], [x for x in range(mer_6_s, mer_6_e+1)])
    return cpgsite2cpginfo


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
