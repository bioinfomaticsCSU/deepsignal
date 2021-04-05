# News
- 2021.03.15: We developed [deepsignal2](https://github.com/PengNi/deepsignal2). Compared to deepsignal, deepsignal2 has much smaller DNN model in size, and slight better performance in 5mCpG detection of human.


# DeepSignal

[![Python](https://img.shields.io/pypi/pyversions/deepsignal)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/deepsignal)](https://pypi.org/project/deepsignal/)
[![GitHub License](https://img.shields.io/github/license/bioinfomaticsCSU/deepsignal)](https://github.com/bioinfomaticsCSU/deepsignal/blob/master/LICENSE)
[![PyPI-Downloads](https://pepy.tech/badge/deepsignal)](https://pepy.tech/project/deepsignal)
[![PyPI-Downloads/m](https://pepy.tech/badge/deepsignal/month)](https://pepy.tech/project/deepsignal/month)

## A deep-learning method for detecting DNA methylation state from Oxford Nanopore sequencing reads.
DeepSignal constructs a BiLSTM+Inception structure to detect DNA methylation state from Nanopore reads. It is
built with **Tensorflow** and **Python 3**.

## Contents
- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Example data](#Example-data)
- [Quick start](#Quick-start)
- [Usage](#Usage)

## Installation
deepsignal is built on Python3. [tombo](https://github.com/nanoporetech/tombo) is required to re-squiggle the raw signals from nanopore reads before running deepsignal.
   - Prerequisites:\
       [Python 3.*](https://www.python.org/)\
       [tensorflow](https://www.tensorflow.org/) (1.8.0<=tensorflow<=1.13.1)\
       [tombo](https://github.com/nanoporetech/tombo)
   - Dependencies:\
       [numpy](http://www.numpy.org/)\
       [h5py](https://github.com/h5py/h5py)\
       [statsmodels](https://github.com/statsmodels/statsmodels/)\
       [scikit-learn](https://scikit-learn.org/stable/)

#### 1. Create an environment
We highly recommend to use a virtual environment for the installation of deepsignal and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):
```bash
# create
conda create -n deepsignalenv python=3.6
# activate
conda activate deepsignalenv
# deactivate
conda deactivate
```
The virtual environment can also be created by using [virtualenv](https://github.com/pypa/virtualenv/).

#### 2. Install deepsignal
- After creating and activating the environment, download and install deepsignal (**lastest version**) from github:
```bash
git clone https://github.com/bioinfomaticsCSU/deepsignal.git
cd deepsignal
python setup.py install
```
**or** install deepsignal using *pip*:
```bash
pip install deepsignal
```

- [tombo](https://github.com/nanoporetech/tombo) is required to be installed in the same environment:
```bash
# install using conda
conda install -c bioconda ont-tombo
# or install using pip
pip install ont-tombo
``` 

- install [tensorflow](https://www.tensorflow.org/)  (version: 1.8.0<=tensorflow<=1.13.1) in the same environment:

```bash
# install using conda
conda install -c anaconda tensorflow==1.13.1
# or install using pip
pip install 'tensorflow==1.13.1'
```
If a GPU-machine is used, install the gpu version of tensorflow. The cpu version is not required:
```bash
# install using conda
conda install -c anaconda tensorflow-gpu==1.13.1
# or install using pip
pip install 'tensorflow-gpu==1.13.1'
```

## Trained models
The models we trained can be downloaded from [google drive](https://drive.google.com/open?id=1zkK8Q1gyfviWWnXUBMcIwEDw3SocJg7P).

Currently we have trained the following models:
   * _model.CpG.R9.4_1D.human_hx1.bn17.sn360.v0.1.7+.tar.gz_: A CpG model trained using HX1 R9.4 1D reads (for **deepsignal>=0.1.7**).
   * _model.CpG.R9.4_1D.human_hx1.bn17.sn360.tar.gz_: A CpG model trained using HX1 R9.4 1D reads (for **deepsignal<=0.1.6**).
   * _model.GATC.R9_2D.tem.puc19.bn17.sn360.tar.gz_: A G*A*TC model trained using pUC19 R9 2D template reads (for **deepsignal<=0.1.6**).

## Example data
The example data can be downloaded from [google drive](https://drive.google.com/open?id=1zkK8Q1gyfviWWnXUBMcIwEDw3SocJg7P).

   * _fast5s.sample.tar.gz_: The data contain ~4000 yeast R9.4 1D reads each with called events (basecalled by Albacore), along with a genome reference.

## Quick start
To call modifications, the raw fast5 files should be basecalled ([Guppy or Albacore](https://nanoporetech.com/community)) and then be re-squiggled by [tombo](https://github.com/nanoporetech/tombo). At last, modifications of specified motifs can be called by deepsignal. The following are commands to call 5mC in CG contexts from the example data:
```bash
# 1. guppy basecall
guppy_basecaller -i fast5s.al -r -s fast5s.al.guppy --config dna_r9.4.1_450bps_hac_prom.cfg
cat fast5s.al.guppy/*.fastq > fast5s.al.guppy.fastq
# 2. tombo resquiggle
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5s.al --fastq-filenames fast5s.al.guppy.fastq --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10
tombo resquiggle fast5s.al GCF_000146045.2_R64_genomic.fna --processes 10 --corrected-group RawGenomeCorrected_001 --basecall-group Basecall_1D_000 --overwrite
# 3. deepsignal call_mods
deepsignal call_mods --input_path fast5s.al/ --model_path model.CpG.R9.4_1D.human_hx1.bn17.sn360.v0.1.7+/bn_17.sn_360.epoch_9.ckpt --result_file fast5s.al.CpG.call_mods.tsv --reference_path GCF_000146045.2_R64_genomic.fna --corrected_group RawGenomeCorrected_001 --nproc 10 --is_gpu no
python /path/to/deepsignal/scripts/call_modification_frequency.py --input_path fast5s.al.CpG.call_mods.tsv --result_file fast5s.al.CpG.call_mods.frequency.tsv
```

## Usage
#### 1. Basecall and re-squiggle
Before run deepsignal, the raw reads should be basecalled ([Guppy or Albacore](https://nanoporetech.com/community)) and then be processed by the *re-squiggle* module of [tombo](https://github.com/nanoporetech/tombo).

Note:
- If the fast5 files are in multi-read FAST5 format, please use _multi_to_single_fast5_ command from the [ont_fast5_api package](https://github.com/nanoporetech/ont_fast5_api) to convert the fast5 files first (Ref to [issue #173](https://github.com/nanoporetech/tombo/issues/173) in [tombo](https://github.com/nanoporetech/tombo)).
```bash
multi_to_single_fast5 -i $multi_read_fast5_dir -s $single_read_fast5_dir -t 30 --recursive
```
- If the basecall results are saved as fastq, run the [*tombo proprecess annotate_raw_with_fastqs*](https://nanoporetech.github.io/tombo/resquiggle.html) command before *re-squiggle*.

For the example data:
```bash
# 1. basecall
guppy_basecaller -i fast5s.al -r -s fast5s.al.guppy --config dna_r9.4.1_450bps_hac_prom.cfg
# 2. proprecess fast5 if basecall results are saved in fastq format
cat fast5s.al.guppy/*.fastq > fast5s.al.guppy.fastq
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5s.al --fastq-filenames fast5s.al.guppy.fastq --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10
# 3. resquiggle, cmd: tombo resquiggle $fast5_dir $reference_fa
tombo resquiggle fast5s.al GCF_000146045.2_R64_genomic.fna --processes 10 --corrected-group RawGenomeCorrected_001 --basecall-group Basecall_1D_000 --overwrite
```

#### 2. extract features
Features of targeted sites can be extracted for training or testing.

For the example data (deepsignal extracts 17-mer-seq and 360-signal features of each CpG motif in reads by default. Note that the value of *--corrected_group* must be the same as that of *--corrected-group* in tombo.):
```bash
deepsignal extract --fast5_dir fast5s.al/ --reference_path GCF_000146045.2_R64_genomic.fna --write_path fast5s.al.CpG.signal_features.17bases.rawsignals_360.tsv --corrected_group RawGenomeCorrected_001 --nproc 10
```

The extracted_features file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **readname**:  the read name
   - **read_strand**:   t/c, template or complement
   - **k_mer**: the sequence around the targeted base
   - **signal_means**:  signal means of each base in the kmer
   - **signal_stds**:   signal stds of each base in the kmer
   - **signal_lens**:   lens of each base in the kmer
   - **cent_signals**:  the central signals of the kmer
   - **methy_label**:   0/1, the label of the targeted base, for training

#### 3. call modifications
The extracted features can be used to call modifications as follows (If a GPU-machine is used, set *--is_gpu* to "yes".):
```bash
# the CpGs are called by using the CpG model of HX1 R9.4 1D
deepsignal call_mods --input_path fast5s.al.CpG.signal_features.17bases.rawsignals_360.tsv --model_path model.CpG.R9.4_1D.human_hx1.bn17.sn360.v0.1.7+/bn_17.sn_360.epoch_9.ckpt --result_file fast5s.al.CpG.call_mods.tsv --nproc 10 --is_gpu no
```

**The modifications can also be called from the fast5 files directly**:
```bash
# use CPU
deepsignal call_mods --input_path fast5s.al/ --model_path model.CpG.R9.4_1D.human_hx1.bn17.sn360.v0.1.7+/bn_17.sn_360.epoch_9.ckpt --result_file fast5s.al.CpG.call_mods.tsv --reference_path GCF_000146045.2_R64_genomic.fna --corrected_group RawGenomeCorrected_001 --nproc 10 --is_gpu no
# or use GPU
CUDA_VISIBLE_DEVICES=0 deepsignal call_mods --input_path fast5s.al/ --model_path model.CpG.R9.4_1D.human_hx1.bn17.sn360.v0.1.7+/bn_17.sn_360.epoch_9.ckpt --result_file fast5s.al.CpG.call_mods.tsv --reference_path GCF_000146045.2_R64_genomic.fna --corrected_group RawGenomeCorrected_001 --nproc 10 --is_gpu yes
```

The modification_call file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **readname**:  the read name
   - **read_strand**:   t/c, template or complement
   - **prob_0**:    [0, 1], the probability of the targeted base predicted as 0 (unmethylated)
   - **prob_1**:    [0, 1], the probability of the targeted base predicted as 1 (methylated)
   - **called_label**:  0/1, unmethylated/methylated
   - **k_mer**:   the kmer around the targeted base

A modification-frequency file can be generated by the script [scripts/call_modification_frequency.py](https://github.com/bioinfomaticsCSU/deepsignal/blob/master/scripts/call_modification_frequency.py) with the modification_call file:
```bash
python /path/to/deepsignal/scripts/call_modification_frequency.py --input_path fast5s.al.CpG.call_mods.tsv --result_file fast5s.al.CpG.call_mods.frequency.tsv --prob_cf 0
```

The modification_frequency file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **prob_0_sum**:    sum of the probabilities of the targeted base predicted as 0 (unmethylated)
   - **prob_1_sum**:    sum of the probabilities of the targeted base predicted as 1 (methylated)
   - **count_modified**:    number of reads in which the targeted base counted as modified
   - **count_unmodified**:  number of reads in which the targeted base counted as unmodified
   - **coverage**:  number of reads aligned to the targeted base
   - **modification_frequency**:    modification frequency
   - **k_mer**:   the kmer around the targeted base

#### 4. train new models
A new model can be trained as follows:
```bash
# need two independent datasets for training and validating
# use deepsignal train -h/--help for more details
deepsignal train --train_file /path/to/train_data/file --valid_file /path/to/valid_data/file --model_dir /dir/to/save/the/new/model
```

Publication
===========
Peng Ni, Neng Huang, Zhi Zhang, De-Peng Wang, Fan Liang, Yu Miao, Chuan-Le Xiao, Feng Luo, and Jianxin Wang, "DeepSignal: detecting DNA methylation state from Nanopore sequencing reads using deep-learning.", Bioinformatics 35, no. 22 (2019): 4586-4595. [doi:10.1093/bioinformatics/btz276](https://doi.org/10.1093/bioinformatics/btz276)

License
=========
Copyright (C) 2018 [Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Feng Luo](mailto:luofeng@clemson.edu), [Peng Ni](mailto:nipeng@csu.edu.cn), [Neng Huang](mailto:huangneng@csu.edu.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

[Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Peng Ni](mailto:nipeng@csu.edu.cn), [Neng Huang](mailto:huangneng@csu.edu.cn),
School of Information Science and Engineering, Central South University, Changsha 410083, China

[Feng Luo](mailto:luofeng@clemson.edu), School of Computing, Clemson University, Clemson, SC 29634, USA
