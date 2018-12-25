#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
fast5dir=$(echo ${1%/})
ref_fp=$2
model_dir=$3
predict_output_file=$4

# extract_features ==========
run_tombo_resquiggle_py="${BASEDIR}/deepsignal/utils/_run_tombo_resquiggle.py"
python $run_tombo_resquiggle_py --fast5_folder $fast5dir --reference_fp $ref_fp --basecall_group Basecall_1D_000 --processes 8

export_tombo_resquiggle_py="${BASEDIR}/deepsignal/utils/_export_tombo_resquiggle.py"
tombo_sl_dir="${fast5dir}.tombo"
python $export_tombo_resquiggle_py --reads_dir $fast5dir -o $tombo_sl_dir --corrected_group RawGenomeCorrected_001 --basecall_subgroup BaseCalled_template

extract_kmer_signal_py="${BASEDIR}/deepsignal/utils/_extract_kmer_signals.py"
signal_feature_file="${fast5dir}.CpG.signal_features.21bases.tsv"
python $extract_kmer_signal_py --fast5_dir $fast5dir --sl_dir $tombo_sl_dir --reference_path $ref_fp --write_path $signal_feature_file --methyed_label 0 --num_bases 10 --corrected_group RawGenomeCorrected_001 --basecall_subgroup BaseCalled_template --motifs CG --methy_loc_in_motif 0

reformat_sample_features_py="${BASEDIR}/deepsignal/utils/_reformat_sample_features.py"
python $reformat_sample_features_py --sf_filepath $signal_feature_file --header yes --bases_num 17 --raw_signals_num 360



# make input data for predicting ==========
signal_feature_reformat_file="${fast5dir}.CpG.signal_features.17bases.rawsignals_360.tsv"
predict_input_dir="${fast5dir}.CpG.signal_features.17bases.rawsignals_360"
python "${BASEDIR}/deepsignal/generate_testing_data.py" -i $signal_feature_reformat_file -o $predict_input_dir -m 60 -b 17 -s 360

# predict =================================
python "${BASEDIR}/deepsignal/predict.py" -i $predict_input_dir -o $model_dir -n 6 -r $predict_output_file -x 17 -y 360 -z 60


rm -r $tombo_sl_dir
rm $signal_feature_file
rm $signal_feature_reformat_file
rm -r $predict_input_dir

