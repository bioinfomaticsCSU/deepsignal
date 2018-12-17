#! /usr/bin/python

alphabeta = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def max_num_bases():
    return 21


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def complement_seq(dnaseq):
    rdnaseq = dnaseq[::-1]
    comseq = ''
    try:
        comseq = ''.join([alphabeta[x] for x in rdnaseq])
    except Exception:
        print('something wrong in the dna sequence.')
    return comseq


def get_refloc_of_methysite_in_motif(seqstr, motif='CG', methyloc_in_motif=0):
    """

    :param seqstr:
    :param motif:
    :param methyloc_in_motif: 0-based
    :return:
    """
    strlen = len(seqstr)
    motiflen = len(motif)
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] == motif:
            sites.append(i+methyloc_in_motif)
    return sites

