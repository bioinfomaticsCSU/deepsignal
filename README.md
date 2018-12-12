# DeepSignal
## DeepSignal:A software to detect DNA modifications from signal-level Oxford Nanopore sequencing data.
Using a deep learning BiLSTM+Inception structure to establish DNA modifications with nanopore sequencing data.
Built with **Tensorflow 1.8** and python 3.

## Contents

- [Install](#install)
    - [Install tensorflow](#instal-tensorflow)
    - [Install using `pip`](#install-using-pip)
    - [Install from Github](#install-from-github)
- [Predict](#predict)
    - [Prepare predict data](#prepare-predict-data)
    - [Run predict](#run-predict)
- [Training](#training)
    - [Prepare training data set](#prepare-training-data-set)
    - [Train a model](#train-a-model)

## Install
### Install tensorflow
We suggest you create a virtual environment to install DeepSignal.
```
conda create -n deepsignal python=3.6
```
then activate virtual environment:
```
source activate deepsignal
```
install tensorflow with GPU-version:
```
conda install tensorflow_gpu==1.8.0
```
or install tensorflow with CPU-version:
```
conda install tensorflow==1.8.0
```
### Install using `pip`
We haven't finish the python package yet.
### Install from Github
You can download the source code from github:
```
git clone https://github.com/bioinfomaticsCSU/deepsignal.git
cd deepsignal
```
## Predict
### Prepare predict data
You can run the file named `generate_testing_data.py` to generate the data to be detected.
```
python generate_testing_data.py -i Input_file -o Output_folder -m Max_read_name_length -b Kmer_size -s Signal_length
```
### Run predict
After preparing the predict data, you can run the file named `predict.py` to get the modification prediction.
```
python predict.py -i Predict_data_file -o Parameter_model_folder -n model_index -r Output_file -x Kmer_size -y Signal_length -z Max_read_name_length
```

## Training
If you have the labeled methylated and non-methylated data, you can train a model to achieve better predict performance on this species
### Prepare training data set
You can run the file named `generate_training_data.py` to generate the training dataset and validate dataset.
```
python generate_training_data.py -i Input_file -o Output_folder -m Max_read_name_length -b Kmer_size -s Signal_length
```
### Train a model
You can run the file named `train.py` to train a new model.
```
python train.py -i Train_data_file -v Validate_data_file -o Output_model_file -g Log_file -e Number_of_epoch -x Kmer_size -y Signal_length -z Max_read_name_length
```