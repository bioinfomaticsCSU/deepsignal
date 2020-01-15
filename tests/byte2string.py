#!/usr/bin/python
import sys
import h5py

reads_group = 'Raw/Reads'
corrected_group = 'RawGenomeCorrected_000'
basecall_subgroup = 'BaseCalled_template'
corrgroup_path = '/'.join(['Analyses', corrected_group])


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
            alignstrand = str(alignment_attrs['mapped_strand'])
            chrom = str(alignment_attrs['mapped_chrom'])
            if chrom.startswith('b'):
                chrom = chrom.split("'")[1]
    else:
        alignstrand = str(alignment_attrs['mapped_strand'])
        chrom = str(alignment_attrs['mapped_chrom'])
    chrom_start = alignment_attrs['mapped_start']

    return strand, alignstrand, chrom, chrom_start


def main():
    fast5_path = "C:\\Users\\npgen\Desktop\\0\\PCT0020_20181004_0004A30B001AE32F_2_A11_D11_sequencing_" \
                 "run_20181004_NPL0294_P2_A11_D11_53765_read_65_ch_335_strand.fast5"
    h5obj = h5py.File(fast5_path, mode='r')

    strand, alignstrand, chrom, chrom_start = _get_alignment_attrs_of_each_strand('/'.join([corrgroup_path,
                                                                                            basecall_subgroup]),
                                                                                  h5obj)

    print(strand, alignstrand, chrom, chrom_start)
    h5obj.close()


if __name__ == '__main__':
    main()
