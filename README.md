# DeepSignal
## DeepSignal:A software to detect DNA modifications from signal-level Oxford Nanopore sequencing data.
Using a deep learning BiLSTM+Inception structure to establish DNA modifications with nanopore sequencing data.
Built with **Tensorflow 1.8** and python 3.

## Contents

- [Install](#install)
    -[Install using 'pip'](#install-using-pip)
    -[Install from Github](#install-from-github)
- [Predict](#predict)
    -[Test run](#test-run)
- [Training](#training)
    -[Prepare training data set](#prepare-training-data-set)
    -[Train a model](#train-a-model)

## Install
### install tensorflow environment
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
### download the source code of DeepSignal
You can download the source code from github:
```
git clone https://github.com/bioinfomaticsCSU/deepsignal.git
cd deepsignal
```



