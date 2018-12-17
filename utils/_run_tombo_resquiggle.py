#! /usr/bin/python
"""
for yeast uv1h 1d reads
"""
import argparse
import os
from subprocess import PIPE, Popen
import shlex
import shutil

abs_dir = os.path.dirname(os.path.realpath(__file__))


# tombo resquiggle ../nanopore_data/yeast/by4741_UV1H/20180122_1224_20180122yeast4741_UV1h/fast5/pass/0/
# ../nanopore_data/yeast/GCF_000146045.2_R64_genomic.fna --processes 20 --corrected-group RawGenomeCorrected_001
# --basecall-group Basecall_1D_000 --overwrite
# >>tombo.20180122_1224_20180122yeast4741_UV1h.fast5.pass.0.log 2>&1 &
# nthread = 30


def tombo_call_in_subdirs(fast5_folder, ref_fa, basecall_group='Basecall_1D_000', processes=45):
    cwd = os.getcwd()
    f5_abs_path = os.path.abspath(fast5_folder)

    if os.path.exists(cwd + '/tombo_logs'):
        print('deleting the previous log direction...')
        shutil.rmtree(cwd + '/tombo_logs')
        print('done')
    os.mkdir(cwd + '/tombo_logs')
    for subdir in os.listdir(f5_abs_path):
        if os.path.isdir(subdir):
            log_file = '.'.join([cwd + '/tombo_logs/tombo',
                                 str(fast5_folder).strip('./').replace('/', '.'),
                                 subdir, 'log'])
            print('running tombo command===========')
            fast5_subdir_folder = '/'.join([f5_abs_path, subdir])
            tombo_call = "bash " + abs_dir + "/test_run_tombo.sh " \
                         "{f5_folder} {ref_fa} {basecall_group} {logfile} {nproc}".format(f5_folder=fast5_subdir_folder,
                                                                                          ref_fa=ref_fa,
                                                                                          basecall_group=basecall_group,
                                                                                          logfile=log_file, nproc=processes)
            print(tombo_call)
            tombo_call_process = Popen(shlex.split(tombo_call), stdout=PIPE, stderr=PIPE)
            out, error = tombo_call_process.communicate()
    log_file = '.'.join([cwd + '/tombo_logs/tombo',
                        str(fast5_folder).strip('./').replace('/', '.'),
                        'log'])
    tombo_call = "bash " + abs_dir + "/test_run_tombo.sh " \
                 "{f5_folder} {ref_fa} {basecall_group} {logfile} {nproc}".format(f5_folder=f5_abs_path,
                                                                                  ref_fa=ref_fa,
                                                                                  basecall_group=basecall_group,
                                                                                  logfile=log_file, nproc=processes)
    tombo_call_process = Popen(shlex.split(tombo_call), stdout=PIPE, stderr=PIPE)
    out, error = tombo_call_process.communicate()
    os.system(' '.join(['rm', '-r', cwd + '/tombo_logs']))


def main():
    parser = argparse.ArgumentParser(description='call nanoraw resquiggle in all subdirs of a folder')
    parser.add_argument('--fast5_folder', type=str, required=True,
                        help='the fast5_folder contains some sub folders')
    # parser.add_argument('--output_folder', type=str, required=True,
    #                     help='the output_folder')
    parser.add_argument('--reference_fp', type=str, required=True,
                        help='the genome reference fp')
    parser.add_argument('--processes', type=int, required=False,
                        default=4,
                        help='number of threads')
    parser.add_argument('--basecall_group', type=str, required=False,
                        default='Basecall_1D_000',
                        help='default Basecall_1D_000; can be Basecall_1D_001 or else')
    args = parser.parse_args()
    f5_dir = args.fast5_folder
    # op_dir = args.output_folder
    ref_fp = args.reference_fp
    processes = args.processes
    basecall_group = args.basecall_group

    tombo_call_in_subdirs(f5_dir, ref_fp, basecall_group, processes)


if __name__ == '__main__':
    main()
