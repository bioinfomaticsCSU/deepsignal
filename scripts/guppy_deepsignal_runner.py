#!/usr/bin/python
"""
install ont_fast5_api, guppy, tombo, deepsignal first
"""
import argparse
import sys
import os
import time


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def _multi_to_single_fast5(args, input_path=None):
    if input_path is None:
        input_path = args.input_path
    input_path = input_path.rstrip("/")
    save_path = input_path + ".single"
    cmd = "multi_to_single_fast5 --input_path {input_path} --save_path {save_path} " \
          "--recursive --threads {threads}".format(input_path=input_path,
                                                   save_path=save_path,
                                                   threads=args.threads)
    sys.stdout.write("cmd: {}\n".format(cmd))
    sys.stdout.flush()
    os.system(cmd)
    return save_path


def _guppy(args, input_path=None):
    if input_path is None:
        input_path = args.input_path
    input_path = input_path.rstrip("/")
    fastq_dir = input_path + ".guppy.fq"
    cmd = "guppy_basecaller -i {input_path} -r -s {fastq_dir} " \
          "--flowcell {flowcell} --kit {kit} --num_callers {num_callers} -x {gpu}".format(input_path=input_path,
                                                                                          fastq_dir=fastq_dir,
                                                                                          flowcell=args.flowcell,
                                                                                          kit=args.kit,
                                                                                          num_callers=args.num_callers,
                                                                                          gpu=args.gpu)
    sys.stdout.write("cmd: {}\n".format(cmd))
    sys.stdout.flush()
    os.system(cmd)
    return fastq_dir, fastq_dir+"/sequencing_summary.txt"


def _tombo_preprocess(args, input_path=None, fastq_dir=None, fastq_summary_txt=None):
    if input_path is None:
        input_path = args.input_path
    if fastq_dir is None:
        fastq_dir = args.fastq_dir
    if fastq_summary_txt is None:
        fastq_summary_txt = args.sequencing_summary_file

    input_path = input_path.rstrip("/")
    fastq_dir = fastq_dir.rstrip("/")
    combined_fastq = fastq_dir + "/combined.fastq"
    cmd1 = "cat {} > {}".format(fastq_dir + "/*.fastq",
                                combined_fastq)
    cmd2 = "tombo preprocess annotate_raw_with_fastqs --fast5-basedir {input_path} " \
           "--fastq-filenames {fastq_file} " \
           "--sequencing-summary-filenames {sequencing_summary} " \
           "--basecall-group {basecall_group} --basecall-subgroup {basecall_subgroup} " \
           "--overwrite --processes {threads}".format(input_path=input_path,
                                                      fastq_file=combined_fastq,
                                                      sequencing_summary=fastq_summary_txt,
                                                      basecall_group=args.basecall_group,
                                                      basecall_subgroup=args.basecall_subgroup,
                                                      threads=args.threads)
    sys.stdout.write("cmd1: {}\n".format(cmd1))
    sys.stdout.write("cmd2: {}\n".format(cmd2))
    sys.stdout.flush()
    os.system(cmd1)
    os.system(cmd2)
    os.remove(combined_fastq)
    return input_path


def _tombo_resquiggle(args, input_path):
    if input_path is None:
        input_path = args.input_path
    input_path = input_path.rstrip("/")
    cmd = "tombo resquiggle {input_path} {ref_fp} " \
          "--processes {threads} --corrected-group {corrected_group} --basecall-group {basecall_group} " \
          "--overwrite --ignore-read-locks".format(input_path=input_path,
                                                   ref_fp=args.ref_fp,
                                                   threads=args.threads,
                                                   corrected_group=args.corrected_group,
                                                   basecall_group=args.basecall_group)
    sys.stdout.write("cmd: {}\n".format(cmd))
    sys.stdout.flush()
    os.system(cmd)
    return input_path


def _deepsignal_call(args, input_path):
    if input_path is None:
        input_path = args.input_path
    input_path = input_path.rstrip("/")
    cmd = "CUDA_VISIBLE_DEVICES={cudanumber} deepsignal call_mods --input_path {input_path} --model_path {model_path} " \
          "--kmer_len {kmer_len} --cent_signals_len {cent_signals_len} --result_file {result_file} " \
          "--corrected_group {corrected_group} --reference_path {ref_fp} " \
          "--motifs {motifs} --mod_loc {mod_loc} " \
          "--nproc {threads} --is_gpu {is_gpu}".format(cudanumber=args.gpu[-1],
                                                       input_path=input_path,
                                                       model_path=args.model_path,
                                                       kmer_len=args.kmer_len,
                                                       cent_signals_len=args.cent_signals_len,
                                                       result_file=args.result_file,
                                                       corrected_group=args.corrected_group,
                                                       ref_fp=args.ref_fp,
                                                       motifs=args.motifs,
                                                       mod_loc=args.mod_loc,
                                                       threads=args.threads,
                                                       is_gpu=args.is_gpu)
    sys.stdout.write("cmd: {}\n".format(cmd))
    sys.stdout.flush()
    os.system(cmd)
    return args.result_file


def run_guppy_deepsignal(args):
    input_path = args.input_path
    fastq_dir, fastq_summary_txt = None, None
    if str2bool(args.is_multi_reads):
        start = time.time()
        sys.stdout.write("[ont_fast5_api]running multi_to_single_fast5================\n")
        input_path = _multi_to_single_fast5(args, input_path)
        sys.stdout.write("[ont_fast5_api]ending multi_to_single_fast5, "
                         "cost {} seconds================\n".format(time.time()-start))
        sys.stdout.flush()
    if str2bool(args.is_guppy):
        start = time.time()
        sys.stdout.write("[guppy]running================\n")
        fastq_dir, fastq_summary_txt = _guppy(args, input_path)
        sys.stdout.write("[guppy]ending, cost {} seconds================\n".format(time.time()-start))
        sys.stdout.flush()
    if str2bool(args.is_tombo):
        start = time.time()
        sys.stdout.write("[tombo]running preprocess================\n")
        _tombo_preprocess(args, input_path, fastq_dir, fastq_summary_txt)
        sys.stdout.write("[tombo]ending preprocess, cost {} seconds================\n".format(time.time()-start))
        start = time.time()
        sys.stdout.write("[tombo]running resquiggle================\n")
        _tombo_resquiggle(args, input_path)
        sys.stdout.write("[tombo]ending resquiggle, cost {} seconds================\n".format(time.time()-start))
        sys.stdout.flush()
    start = time.time()
    sys.stdout.write("[deepsignal]running================\n")
    _deepsignal_call(args, input_path)
    sys.stdout.write("[deepsignal]ending, cost {} seconds================\n".format(time.time()-start))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser("guppy_basecall+deepsignal_mod_call script\n"
                                     "Input: Nanopore raw reads\n"
                                     "Output: deepsignal mods call")

    parser.add_argument("--threads", type=int, default=10,
                        required=False,
                        help="number of threads to use, default 10")

    pinput = parser.add_argument_group("INPUT")
    pinput.add_argument("--input_path", type=str, default=None,
                        required=False,
                        help="dir contains fast5 reads")
    pinput.add_argument("--is_multi_reads", type=str, default="yes",
                        choices=["yes", "no"],
                        required=True,
                        help="are the input reads in multi-format, default yes")
    pinput.add_argument("--ref_fp", type=str, default=None,
                        required=True,
                        help="reference file path")

    pguppy = parser.add_argument_group("guppy")
    pguppy.add_argument("--is_guppy", type=str, default="no",
                        choices=["yes", "no"], required=False,
                        help="is need to basecall by guppy, default no")
    pguppy.add_argument("--flowcell", type=str, default="FLO-PRO002",
                        required=False,
                        help="flowcell for guppy, default FLO-PRO002")
    pguppy.add_argument("--kit", type=str, default="SQK-LSK109",
                        required=False,
                        help="kit for guppy, default SQK-LSK109")
    pguppy.add_argument("--num_callers", type=int, default=10,
                        required=False,
                        help="num_callers for guppy, default 10")
    pguppy.add_argument("--gpu", type=str, default="cuda:0",
                        required=False,
                        help="gpu for guppy, default cuda:0")

    ptombo = parser.add_argument_group("tombo")
    ptombo.add_argument("--is_tombo", type=str, default="no",
                        choices=["yes", "no"], required=False,
                        help="is need to resquiggle by tombo, default no")
    ptombo.add_argument("--basecall_group", type=str, default="Basecall_1D_000",
                        required=False,
                        help="basecall_group, default Basecall_1D_000")
    ptombo.add_argument("--basecall_subgroup", type=str, default="BaseCalled_template",
                        required=False,
                        help="basecall_subgroup, default BaseCalled_template")

    ptombo_pre = parser.add_argument_group("tombo preprocess")
    ptombo_pre.add_argument("--fastq_dir", type=str, default=None,
                            required=False,
                            help="fastq directory contains guppy basecall results")
    ptombo_pre.add_argument("--sequencing_summary_file", type=str, default=None,
                            required=False,
                            help="sequencing_summary.txt file path")

    ptombo_res = parser.add_argument_group("tombo resquiggle")
    ptombo_res.add_argument("--corrected_group", type=str, default="RawGenomeCorrected_000",
                            required=False,
                            help="corrected_group for tombo resquiggle, this arg is used by deepsignal too, "
                                 "default RawGenomeCorrected_000")

    pdeepsignal = parser.add_argument_group("deepsignal")
    pdeepsignal.add_argument("--model_path", "-m", action="store", type=str, required=True,
                             help="file path of the trained model (.ckpt)")
    pdeepsignal.add_argument("--kmer_len", "-x", action="store", default=17, type=int, required=False,
                             help="base num of the kmer, default 17")
    pdeepsignal.add_argument("--cent_signals_len", "-y", action="store", default=360, type=int, required=False,
                             help="the number of central signals of the kmer to be used, default 360")
    pdeepsignal.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                             action="store", help="batch size, default 512")
    pdeepsignal.add_argument("--learning_rate", "-l", default=0.001, type=float, required=False,
                             action="store", help="init learning rate, default 0.001")
    pdeepsignal.add_argument("--class_num", "-c", action="store", default=2, type=int, required=False,
                             help="class num, default 2")

    pdeepsignal.add_argument("--result_file", "-o", action="store", type=str, required=True,
                             help="the file path to save the predicted result")

    pdeepsignal.add_argument("--motifs", action="store", type=str,
                             required=False, default='CG',
                             help='motif seq to be extracted, default: CG. '
                                  'can be multi motifs splited by comma '
                                  '(no space allowed in the input str), '
                                  'or use IUPAC alphabet, '
                                  'the mod_loc of all motifs must be '
                                  'the same')
    pdeepsignal.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                             help='0-based location of the targeted base in the motif, default 0')

    pdeepsignal.add_argument("--is_gpu", action="store", type=str, default="yes", required=False,
                             choices=["yes", "no"], help="use gpu for tensorflow or not, default no. "
                                                         "If you're using a gpu machine, please set to yes. "
                                                         "Note that when is_gpu is yes, --nproc is not valid "
                                                         "to tensorflow.")

    args = parser.parse_args()
    start = time.time()
    sys.stdout.write("[main]start============================================\n")
    run_guppy_deepsignal(args)
    sys.stdout.write("[main]end, cost {} seconds=============================\n".format(time.time()-start))
    sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(main())
