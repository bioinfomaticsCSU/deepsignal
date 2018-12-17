# DeepSignal
## DeepSignal:A software to detect DNA methylation state from Oxford Nanopore sequencing reads.
Using a deep learning BiLSTM+Inception structure to detecte DNA modification state with nanopore sequencing data.
Built with **Tensorflow 1.8** and python 3.

## Contents

- [Install](#install)
    - [Install Dependencies](#Install-Dependencies)
    - [Install DeepSignal from Github](#Install-DeepSignal-from-Github)
- [Extract features](#Extract-features)
- [Predict](#predict)
    - [Prepare predict data](#prepare-predict-data)
    - [Run predict](#run-predict)
- [Training](#training)
    - [Prepare training data set](#prepare-training-data-set)
    - [Train a model](#train-a-model)
- [Example](#Example)

## Install
### Install Dependencies
We suggest you create a virtual environment to install DeepSignal and its dependencies.
#### Dependencies
   - [Python 3.*](https://www.python.org/)
   - Python packages:\
       [h5py](https://github.com/h5py/h5py)\
       [tombo](https://github.com/nanoporetech/tombo)\
       [statsmodels](https://github.com/statsmodels/statsmodels/)\
       [sklearn](https://scikit-learn.org/stable/)\
       [tensorflow v1.8.0](https://www.tensorflow.org/)

#### Install using conda
```bash
conda create -n deepsignal python=3.6
```
then activate virtual environment:
```bash
source activate deepsignal
```
install the dependencies:
```bash
conda install h5py
conda install -c bioconda ont-tombo
conda install statsmodels
conda install -c anaconda scikit-learn
conda install tensorflow_gpu==1.8.0
```
or install tensorflow with CPU-version:
```bash
conda install tensorflow==1.8.0
```
A virtual environment can also be created using [*virtualenv*](https://github.com/pypa/virtualenv/).

#### Install using `pip`
```bash
# in the environment of python3
pip install numpy
pip install ont-tombo[full]
pip install statsmodels
pip install sklearn
pip install 'tensorflow==1.8.0'
```

### Install DeepSignal from Github
You can download the source code from github:
```
git clone https://github.com/bioinfomaticsCSU/deepsignal.git
cd deepsignal
```

## Extract features
After basecalling, the signal features can be extracted. Please refer to [Example](#Example) for specific pipelines.

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

## Train
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

## Example
The models we trained and the example data can be downloaded from [here](http://bioinformatics.csu.edu.cn/).

* The model is CpG_model trained using HX1 R9.4 1D reads.
* The example data is ~4000 reads of yeast R9.4 1D reads, along with a genome reference.

After downloading, the script *pipeline_demo.sh* can be used test the data:
```bash
chmod +x /path/to/pipeline_demo.sh
/path/to/pipeline_demo.sh /path/to/fast5_folder /path/to/genome_ref.fa /path/to/model_folder /path/to/output_result
```


