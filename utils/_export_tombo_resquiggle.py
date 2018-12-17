"""
to simulate chiron export function for tombo re-squiggle result
"""
import h5py
import numpy as np
import argparse
import os
import shutil


def get_label_raw(fast5_fn, correct_group, correct_subgroup):
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        raw_dat = list(fast5_data['/Raw/Reads/'].values())[0]
        # raw_attrs = raw_dat.attrs
        raw_dat = raw_dat['Signal'].value
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' + 'new segments cannot be identified.')

    # Get Events
    try:
        event = fast5_data['/Analyses/'+correct_group + '/'+correct_subgroup+'/Events']
        corr_attrs = dict(list(event.attrs.items()))
    except:
        raise RuntimeError('events not found.')
    starts = list()
    lengths = list()
    read_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']
    # print(event)
    # print('read_start_rel_to_raw: ',read_start_rel_to_raw)
    starts = list(map(lambda x: x+read_start_rel_to_raw, event['start']))
    lengths = event['length'].astype(np.int)
    base = [x.decode("UTF-8") for x in event['base']]
    assert len(starts) == len(lengths)
    assert len(lengths) == len(base)
    events = list(zip(starts, lengths, base))
    return raw_dat, events


def run(argv):
    dircount = 0
    count = 0
    error = 0
    if os.path.exists(argv.output_dir):
        print('deleting the previous output direction...')
        shutil.rmtree(argv.output_dir)
        print('done')
    os.makedirs(argv.output_dir)
    for it in os.listdir(argv.reads_dir):
        if os.path.isdir(argv.reads_dir+'/'+it):
            out_signal_path = argv.output_dir+'/'+it
            out_raw_path = argv.output_dir+'/'+it
            assert out_signal_path == out_raw_path
            os.mkdir(out_signal_path)
            for f in os.listdir(argv.reads_dir+'/'+it):
                f5file = argv.reads_dir+'/'+it+'/'+f
                if not f5file.endswith('.fast5'):
                    continue
                try:
                    raw, evt = get_label_raw(f5file, argv.corrected_group, argv.basecall_subgroup)
                except:
                    error += 1
                    continue
                out_path_s = out_signal_path+'/'+os.path.splitext(os.path.basename(f))[0]+'.label'
                out_path_r = out_raw_path+'/'+os.path.splitext(os.path.basename(f))[0]+'.signal'
                fs = open(out_path_s, 'w')
                for v in evt:
                    fs.write(str(v[0])+' '+str(v[0]+v[1])+' '+str(v[2])+'\n')
                fs.close()
                fr = open(out_path_r, 'w')
                for v in raw:
                    fr.write(str(int(v))+' ')
                fr.close()
                count += 1
            dircount += 1
            print("{} dir {} done..".format(dircount, it))
    print('success file number: ', count)
    print('defeat file number: ', error)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--reads_dir", help="the directory of reads fast5 files",
                        type=str, required=True)
    parser.add_argument('-o', "--output_dir", help="output dir for .signal and .label files",
                        type=str, required=True)
    parser.add_argument('--corrected_group', type=str, required=False,
                        default='RawGenomeCorrected_001',
                        help='default RawGenomeCorrected_001')
    parser.add_argument('--basecall_subgroup', type=str, required=False,
                        default='BaseCalled_template',
                        help='default BaseCalled_template; BaseCalled_complement')

    argv = parser.parse_args()
    run(argv)


if __name__ == '__main__':
    main()
    pass
