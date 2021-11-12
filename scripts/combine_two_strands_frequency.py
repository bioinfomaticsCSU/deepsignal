#! /usr/bin/python
import argparse
import os


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


def combine_fb_of_freqtxt(report_fp, cgposes):
    pos2info = {}
    for cgpos in cgposes:
        pos2info[cgpos] = [0.0, 0.0, 0, 0, 0, 0.0, '-']
    with open(report_fp, "r") as rf:
        # next(rf)
        for line in rf:
            words = line.strip().split('\t')
            keytmp = (words[0], int(words[1]))
            if words[2] == '-':
                keytmp = (words[0], int(words[1]) - 1)
                if keytmp not in cgposes:
                    print("{}, not in selected motif poses of the genome".format(words))
                    continue
            else:
                if keytmp not in cgposes:
                    print("{}, not in selected motif poses of the genome".format(words))
                    continue
                pos2info[keytmp][6] = words[10]
            prob0, prob1, met, unmet, coverage = float(words[4]), float(words[5]), \
                int(words[6]), int(words[7]), int(words[8])
            pos2info[keytmp][0] += prob0
            pos2info[keytmp][1] += prob1
            pos2info[keytmp][2] += met
            pos2info[keytmp][3] += unmet
            pos2info[keytmp][4] += coverage
    for cgpos in list(pos2info.keys()):
        if pos2info[cgpos][4] == 0:
            del pos2info[cgpos]
        else:
            pos2info[cgpos][5] = float(pos2info[cgpos][2]) / pos2info[cgpos][4]
    mposinfo = []
    for cgpos in pos2info.keys():
        mposinfo.append(list(cgpos) + ['+', cgpos[1]] + pos2info[cgpos])
    mposinfo = sorted(mposinfo, key=lambda x: (x[0], x[1]))
    return mposinfo


def combine_fb_of_bed(report_fp, cgposes):
    pos2info = {}
    for cgpos in cgposes:
        pos2info[cgpos] = [0, 0.0, 0.0]  # coverage, met, rmet
    with open(report_fp, "r") as rf:
        # next(rf)
        for line in rf:
            words = line.strip().split('\t')
            keytmp = (words[0], int(words[1]))
            if words[5] == '-':
                keytmp = (words[0], int(words[1]) - 1)
            if keytmp not in cgposes:
                print("{}, not in selected motif poses of the genome".format(words))
                continue
            coverage, met = int(words[9]), float(words[10]) / 100 * int(words[9])
            try:
                pos2info[keytmp][0] += coverage
                pos2info[keytmp][1] += met
            except KeyError:
                pass
    for cgpos in list(pos2info.keys()):
        if pos2info[cgpos][0] == 0:
            del pos2info[cgpos]
        else:
            pos2info[cgpos][2] = float(pos2info[cgpos][1]) / pos2info[cgpos][0]
    mposinfo = []
    for cgpos in pos2info.keys():
        chrom, fpos = cgpos[0], cgpos[1]
        mposinfo.append([chrom, fpos, fpos+1, ".", pos2info[cgpos][0], "+",
                         fpos, fpos+1, "0,0,0", pos2info[cgpos][0],
                         int(round(pos2info[cgpos][2], 2) * 100)])
    mposinfo = sorted(mposinfo, key=lambda x: (x[0], x[1]))
    return mposinfo


def write_mpos2covinfo_deep(mclist, reportfp):
    with open(reportfp, 'w') as wf:
        # wf.write('\t'.join(['chromosome', 'pos', 'strand', 'pos_in_strand', 'prob0', 'prob1',
        #                     'met', 'unmet', 'coverage', 'Rmet', 'kmer']) + '\n')
        for mctmp in mclist:
            wf.write('\t'.join(list(map(str, list(mctmp)))) + '\n')
    return mclist


def main():
    parser = argparse.ArgumentParser("combine modification_frequency of CG in forward and backward strand")
    parser.add_argument("--frequency_fp", help="the call_modification_frequency file path, "
                                               "in freq.txt or .bed format",
                        type=str, required=True)
    parser.add_argument('-r', "--ref_fp", help="the file path of genome reference",
                        type=str, required=True)
    parser.add_argument('--contig', type=str, required=False, default='',
                        help='contig name need to be processed, if default, then all contigs are used')
    # parser.add_argument('--motif', type=str, required=False, default='CG',
    #                     help='targeted motif, must be palindrome, default CG')
    # parser.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
    #                     help='0-based location of the targeted base in the motif, default 0')
    argv = parser.parse_args()

    report_fp = argv.frequency_fp
    ref_fp = argv.ref_fp
    contign = argv.contig
    # motif = argv.motif
    # mod_loc = argv.mod_loc
    motif = "CG"
    mod_loc = 0

    print('start to get genome reference info..')
    refseq = DNAReference(ref_fp)
    contigname2contigseq = refseq.getcontigs()
    del refseq

    print('start to get motif poses in genome reference..')
    contig_cg_poses = set()
    if contign == '':
        for cgname in contigname2contigseq.keys():
            fcseq = contigname2contigseq[cgname]
            fposes = get_refloc_of_methysite_in_motif(fcseq, motif, mod_loc)
            for fpos in fposes:
                contig_cg_poses.add((cgname, fpos))
    else:
        fcseq = contigname2contigseq[contign]
        fposes = get_refloc_of_methysite_in_motif(fcseq, motif, mod_loc)
        for fpos in fposes:
            contig_cg_poses.add((contign, fpos))

    print('start to combine forward backward strands..')
    fname, fext = os.path.splitext(report_fp)
    wfp = fname + '.fb_combined' + fext

    if not str(report_fp).lower().endswith(".bed"):
        mposinfo = combine_fb_of_freqtxt(report_fp, contig_cg_poses)
    else:
        mposinfo = combine_fb_of_bed(report_fp, contig_cg_poses)
    write_mpos2covinfo_deep(mposinfo, wfp)


if __name__ == '__main__':
    main()
