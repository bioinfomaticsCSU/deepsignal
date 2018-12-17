#!/usr/bin/env bash

f5_dir=$1
ref_fp=$2
basecall_group=$3
log_file=$4
nproc=$5
tombo resquiggle $f5_dir $ref_fp --processes $nproc --corrected-group RawGenomeCorrected_001 --basecall-group $basecall_group --overwrite >> $log_file 2>&1