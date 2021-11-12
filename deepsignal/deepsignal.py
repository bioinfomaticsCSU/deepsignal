#! /usr/bin/env python

from __future__ import absolute_import

import argparse

from .utils.process_utils import str2bool
from .utils.process_utils import display_args


def main_extraction(args):
    from .extract_features import extract_features

    display_args(args)

    fast5_dir = args.fast5_dir
    is_recursive = str2bool(args.recursively)

    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    normalize_method = args.normalize_method

    reference_path = args.reference_path
    is_dna = str2bool(args.is_dna)
    write_path = args.write_path
    w_is_dir = str2bool(args.w_is_dir)
    w_batch_num = args.w_batch_num

    kmer_len = args.kmer_len
    cent_signals_num = args.cent_signals_len
    motifs = args.motifs
    mod_loc = args.mod_loc
    methy_label = args.methy_label
    position_file = args.positions

    nproc = args.nproc
    f5_batch_num = args.f5_batch_num

    extract_features(fast5_dir, is_recursive, reference_path, is_dna,
                     f5_batch_num, write_path, nproc, corrected_group, basecall_subgroup,
                     normalize_method, motifs, mod_loc, kmer_len, cent_signals_num, methy_label,
                     position_file, w_is_dir, w_batch_num)


def main_call_mods(args):
    from .call_modifications import call_mods

    display_args(args)

    input_path = args.input_path

    model_path = args.model_path
    result_file = args.result_file

    is_cnn = str2bool(args.is_cnn)
    is_base = str2bool(args.is_base)
    is_rnn = str2bool(args.is_rnn)

    kmer_len = args.kmer_len
    cent_signals_len = args.cent_signals_len

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    class_num = args.class_num

    nproc = args.nproc
    is_gpu = str2bool(args.is_gpu)

    # for FAST5_EXTRACTION
    is_recursive = str2bool(args.recursively)
    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    reference_path = args.reference_path
    is_dna = str2bool(args.is_dna)
    normalize_method = args.normalize_method
    motifs = args.motifs
    mod_loc = args.mod_loc
    # methy_label = args.methy_label
    methy_label = 1
    f5_batch_num = args.f5_batch_num
    position_file = args.positions

    f5_args = (is_recursive, corrected_group, basecall_subgroup, reference_path, is_dna,
               normalize_method, motifs, mod_loc, methy_label, f5_batch_num, position_file)

    call_mods(input_path, model_path, result_file, kmer_len, cent_signals_len,
              batch_size, learning_rate, class_num, nproc, is_gpu, is_rnn, is_base, is_cnn,
              f5_args)


def main_train(args):
    from .train_model import train

    display_args(args)

    train_file = args.train_file
    valid_file = args.valid_file
    is_binary = str2bool(args.is_binary)

    model_dir = args.model_dir
    log_dir = args.log_dir

    is_cnn = str2bool(args.is_cnn)
    is_base = str2bool(args.is_base)
    is_rnn = str2bool(args.is_rnn)

    kmer_len = args.kmer_len
    cent_signals_len = args.cent_signals_len
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    class_num = args.class_num
    keep_prob = args.keep_prob
    max_epoch_num = args.max_epoch_num
    min_epoch_num = args.min_epoch_num
    display_step = args.display_step
    pos_weight = args.pos_weight

    train(train_file, valid_file, model_dir, log_dir, kmer_len, cent_signals_len,
          batch_size, learning_rate, decay_rate, class_num, keep_prob, max_epoch_num,
          min_epoch_num, display_step, pos_weight, is_binary, is_rnn, is_base, is_cnn)


def main_denoise(args):
    from .denoise import denoise

    display_args(args)
    denoise(args)


def main():
    parser = argparse.ArgumentParser(prog='deepsignal',
                                     description="detecting base modifications from Nanopore sequencing reads, "
                                                 "deepsignal contains four modules: \n"
                                                 "\t%(prog)s extract: extract features from corrected (tombo) "
                                                 "fast5s for training or testing\n"
                                                 "\t%(prog)s call_mods: call modifications\n"
                                                 "\t%(prog)s train: train a model, need two independent "
                                                 "datasets for training and validating\n"
                                                 "\t%(prog)s denoise: denoise training samples by deep-learning, "
                                                 "filter false positive samples",
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(title="modules", help='deepsignal modules, use -h/--help for help')
    sub_extract = subparsers.add_parser("extract", description="extract features from corrected (tombo) fast5s for "
                                                               "training or testing."
                                                               "\nIt is suggested that running this module 1 flowcell "
                                                               "a time, or a group of flowcells a time, if the whole "
                                                               "data is extremely large.")
    sub_call_mods = subparsers.add_parser("call_mods", description="call modifications")
    sub_train = subparsers.add_parser("train", description="train a model, need two independent datasets for training "
                                                           "and validating")
    sub_denoise = subparsers.add_parser("denoise", description="denoise training samples by deep-learning, "
                                                               "filter false positive samples")

    # sub_extract ============================================================================
    se_input = sub_extract.add_argument_group("INPUT")
    se_input.add_argument("--fast5_dir", "-i", action="store", type=str,
                          required=True,
                          help="the directory of fast5 files")
    se_input.add_argument("--recursively", "-r", action="store", type=str, required=False,
                          default='yes',
                          help='is to find fast5 files from fast5_dir recursively. '
                               'default true, t, yes, 1')
    se_input.add_argument("--corrected_group", action="store", type=str, required=False,
                          default='RawGenomeCorrected_000',
                          help='the corrected_group of fast5 files after '
                               'tombo re-squiggle. default RawGenomeCorrected_000')
    se_input.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                          default='BaseCalled_template',
                          help='the corrected subgroup of fast5 files. default BaseCalled_template')
    se_input.add_argument("--reference_path", action="store",
                          type=str, required=True,
                          help="the reference file to be used, usually is a .fa file")
    se_input.add_argument("--is_dna", action="store", type=str, required=False,
                          default='yes',
                          help='whether the fast5 files from DNA sample or not. '
                               'default true, t, yes, 1. '
                               'set this option to no/false/0 if '
                               'the fast5 files are from RNA sample.')

    se_extraction = sub_extract.add_argument_group("EXTRACTION")
    se_extraction.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                               default="mad", required=False,
                               help="the way for normalizing signals in read level. "
                                    "mad or zscore, default mad")
    se_extraction.add_argument("--methy_label", action="store", type=int,
                               choices=[1, 0], required=False, default=1,
                               help="the label of the interested modified bases, this is for training."
                                    " 0 or 1, default 1")
    se_extraction.add_argument("--kmer_len", action="store",
                               type=int, required=False, default=17,
                               help="len of kmer. default 17")
    se_extraction.add_argument("--cent_signals_len", action="store",
                               type=int, required=False, default=360,
                               help="the number of signals to be used in deepsignal, default 360")
    se_extraction.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    se_extraction.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    # se_extraction.add_argument("--region", action="store", type=str,
    #                            required=False, default=None,
    #                            help="region of interest, e.g.: chr1:0-10000, default None, "
    #                                 "for the whole region")
    se_extraction.add_argument("--positions", action="store", type=str,
                               required=False, default=None,
                               help="file with a list of positions interested (must be formatted as tab-separated file"
                                    " with chromosome, position (in fwd strand), and strand. default None")

    se_output = sub_extract.add_argument_group("OUTPUT")
    se_output.add_argument("--write_path", "-o", action="store",
                           type=str, required=True,
                           help='file path to save the features')
    se_output.add_argument("--w_is_dir", action="store",
                           type=str, required=False, default="no",
                           help='if using a dir to save features into multiple files')
    se_output.add_argument("--w_batch_num", action="store",
                           type=int, required=False, default=200,
                           help='features batch num to save in a single writed file when --is_dir is true')

    sub_extract.add_argument("--nproc", "-p", action="store", type=int, default=1,
                             required=False,
                             help="number of processes to be used, default 1")
    sub_extract.add_argument("--f5_batch_num", action="store", type=int, default=100,
                             required=False,
                             help="number of files to be processed by each process one time, default 100")

    sub_extract.set_defaults(func=main_extraction)

    # sub_call_mods =============================================================================================
    sc_input = sub_call_mods.add_argument_group("INPUT")
    sc_input.add_argument("--input_path", "-i", action="store", type=str,
                          required=True,
                          help="the input path, can be a signal_feature file from extract_features.py, "
                               "or a directory of fast5 files. If a directory of fast5 files is provided, "
                               "args in FAST5_EXTRACTION should (reference_path must) be provided.")

    sc_call = sub_call_mods.add_argument_group("CALL")
    sc_call.add_argument("--model_path", "-m", action="store", type=str, required=True,
                         help="file path of the trained model (.ckpt)")

    sc_call.add_argument('--is_cnn', type=str, default='yes', required=False,
                         help="dose the used model contain inception module?")
    sc_call.add_argument('--is_rnn', type=str, default='yes', required=False,
                         help="dose the used model contain BiLSTM module?")
    sc_call.add_argument('--is_base', type=str, default='yes', required=False,
                         help="dose the BiLSTM module of the used model take base features as input?")

    sc_call.add_argument("--kmer_len", "-x", action="store", default=17, type=int, required=False,
                         help="base num of the kmer, default 17")
    sc_call.add_argument("--cent_signals_len", "-y", action="store", default=360, type=int, required=False,
                         help="the number of central signals of the kmer to be used, default 360")
    sc_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                         action="store", help="batch size, default 512")
    sc_call.add_argument("--learning_rate", "-l", default=0.001, type=float, required=False,
                         action="store", help="init learning rate, default 0.001")
    sc_call.add_argument("--class_num", "-c", action="store", default=2, type=int, required=False,
                         help="class num, default 2")

    sc_output = sub_call_mods.add_argument_group("OUTPUT")
    sc_output.add_argument("--result_file", "-o", action="store", type=str, required=True,
                           help="the file path to save the predicted result")

    sc_f5 = sub_call_mods.add_argument_group("FAST5_EXTRACTION")
    sc_f5.add_argument("--recursively", "-r", action="store", type=str, required=False,
                       default='yes', help='is to find fast5 files from fast5 dir recursively. '
                                           'default true, t, yes, 1')
    sc_f5.add_argument("--corrected_group", action="store", type=str, required=False,
                       default='RawGenomeCorrected_000',
                       help='the corrected_group of fast5 files after '
                            'tombo re-squiggle. default RawGenomeCorrected_000')
    sc_f5.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                       default='BaseCalled_template',
                       help='the corrected subgroup of fast5 files. default BaseCalled_template')
    sc_f5.add_argument("--reference_path", action="store",
                       type=str, required=False,
                       help="the reference file to be used, usually is a .fa file")
    sc_f5.add_argument("--is_dna", action="store", type=str, required=False,
                       default='yes',
                       help='whether the fast5 files from DNA sample or not. '
                            'default true, t, yes, 1. '
                            'setting this option to no/false/0 means '
                            'the fast5 files are from RNA sample.')
    sc_f5.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                       default="mad", required=False,
                       help="the way for normalizing signals in read level. "
                            "mad or zscore, default mad")
    # sc_f5.add_argument("--methy_label", action="store", type=int,
    #                    choices=[1, 0], required=False, default=1,
    #                    help="the label of the interested modified bases, this is for training."
    #                         " 0 or 1, default 1")
    sc_f5.add_argument("--motifs", action="store", type=str,
                       required=False, default='CG',
                       help='motif seq to be extracted, default: CG. '
                            'can be multi motifs splited by comma '
                            '(no space allowed in the input str), '
                            'or use IUPAC alphabet, '
                            'the mod_loc of all motifs must be '
                            'the same')
    sc_f5.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                       help='0-based location of the targeted base in the motif, default 0')
    sc_f5.add_argument("--f5_batch_num", action="store", type=int, default=100,
                       required=False,
                       help="number of files to be processed by each process one time, default 100")
    sc_f5.add_argument("--positions", action="store", type=str,
                       required=False, default=None,
                       help="file with a list of positions interested (must be formatted as tab-separated file"
                            " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                            "need to be set. --positions is used to narrow down the range of the trageted "
                            "motif locs. default None")

    sub_call_mods.add_argument("--nproc", "-p", action="store", type=int, default=1,
                               required=False, help="number of processes to be used, default 1.")
    sub_call_mods.add_argument("--is_gpu", action="store", type=str, default="no", required=False,
                               choices=["yes", "no"], help="use gpu for tensorflow or not, default no. "
                                                           "If you're using a gpu machine, please set to yes. "
                                                           "Note that when is_gpu is yes, --nproc is not valid "
                                                           "to tensorflow.")

    sub_call_mods.set_defaults(func=main_call_mods)

    # sub_train =====================================================================================
    st_input = sub_train.add_argument_group("INPUT")
    st_input.add_argument("--train_file", action="store", type=str, required=True,
                          help="file contains samples for training, from extract_features.py. "
                               "The file should contain shuffled positive and negative samples. "
                               "For CpG, Up to 20m (~10m positive and ~10m negative) samples are sufficient.")
    st_input.add_argument("--valid_file", action="store", type=str, required=True,
                          help="file contains samples for testing, from extract_features.py. "
                               "The file should contain shuffled positive and negative samples. "
                               "For CpG, 10k (~5k positive and ~5k negative) samples are sufficient.")
    st_input.add_argument("--is_binary", action="store", type=str, required=False,
                          default="no", choices=["yes", "no"],
                          help="are the train_file and valid_file in binary format or not? "
                               "'yes' or 'no', default no. "
                               "(for binary format, see scripts/generate_binary_feature_file.py)")

    st_output = sub_train.add_argument_group("OUTPUT")
    st_output.add_argument("--model_dir", "-o", action="store", type=str, required=True,
                           help="directory for saving the trained model")
    st_output.add_argument("--log_dir", "-g", action="store", type=str, required=False,
                           default=None,
                           help="directory for saving the training log")

    st_train = sub_train.add_argument_group("TRAIN")
    st_train.add_argument('--is_cnn', type=str, default='yes', required=False,
                          help="using inception module of deepsignal or not")
    st_train.add_argument('--is_base', type=str, default='yes', required=False,
                          help="using base features in BiLSTM module or not")
    st_train.add_argument('--is_rnn', type=str, default='yes', required=False,
                          help="using BiLSTM module of deepsignal or not")

    st_train.add_argument("--kmer_len", "-x", action="store", default=17, type=int, required=False,
                          help="base num of the kmer, default 17")
    st_train.add_argument("--cent_signals_len", "-y", action="store", default=360, type=int, required=False,
                          help="the number of central signals of the kmer to be used, default 360")

    st_train.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                          action="store", help="batch size, default 512")
    st_train.add_argument("--learning_rate", "-l", default=0.001, type=float, required=False,
                          action="store", help="init learning rate, default 0.001")
    st_train.add_argument("--decay_rate", "-d", default=0.1, type=float, required=False,
                          action="store", help="decay rate, default 0.1")
    st_train.add_argument("--class_num", "-c", action="store", default=2, type=int, required=False,
                          help="class num, default 2")
    st_train.add_argument("--keep_prob", action="store", default=0.5, type=float,
                          required=False, help="keep prob, default 0.5")
    st_train.add_argument("--max_epoch_num", action="store", default=10, type=int,
                          required=False, help="max epoch num, default 10")
    st_train.add_argument("--min_epoch_num", action="store", default=5, type=int,
                          required=False, help="min epoch num, default 5")
    st_train.add_argument("--display_step", action="store", default=100, type=int,
                          required=False, help="display step, default 100")
    st_train.add_argument("--pos_weight", action="store", default=1.0, type=float,
                          required=False, help="pos_weight in loss function: "
                                               "tf.nn.weighted_cross_entropy_with_logits, "
                                               "for imbalanced training samples, default 1. "
                                               "If |pos samples| : |neg samples| = 1:3, set pos_weight to 3.")

    sub_train.set_defaults(func=main_train)

    # sub_denoise =====================================================================================
    sd_input = sub_denoise.add_argument_group("INPUT")
    sd_input.add_argument('--train_file', type=str, required=True)

    sd_train = sub_denoise.add_argument_group("TRAIN")
    sd_train.add_argument('--model_prefix', type=str, default="model",
                          required=False)

    sd_train.add_argument('--is_cnn', type=str, default='no', required=False)
    sd_train.add_argument('--is_base', type=str, default='no', required=False)
    sd_train.add_argument('--is_rnn', type=str, default='yes', required=False)

    sd_train.add_argument('--seq_len', type=int, default=17, required=False)
    sd_train.add_argument('--cent_signals_len', type=int, default=360, required=False)
    sd_train.add_argument('--layer_num', type=int, default=3, required=False)
    sd_train.add_argument('--class_num', type=int, default=2, required=False)
    sd_train.add_argument('--batch_size', type=int, default=512, required=False)
    sd_train.add_argument('--lr', type=float, default=0.001, required=False,
                          help="learning rate")
    sd_train.add_argument('--decay_rate', type=float, default=0.1, required=False)
    sd_train.add_argument('--keep_prob', action="store", default=0.5, type=float,
                          required=False, help="keep prob, default 0.5")
    sd_train.add_argument('--pos_weight', type=float, default=1.0, required=False)

    sd_train.add_argument('--iterations', type=int, default=6, required=False)
    sd_train.add_argument('--epoch_num', type=int, default=5, required=False)
    sd_train.add_argument('--step_interval', type=int, default=100, required=False)
    sd_train.add_argument('--rounds', type=int, default=5, required=False)
    sd_train.add_argument("--score_cf", type=float, default=0.5,
                          required=False,
                          help="score cutoff")

    sub_denoise.set_defaults(func=main_denoise)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
