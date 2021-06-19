# Text Binary Classification Example

Methods: CNN, LSTM, Transformer, DPCNN

Two Levels: Char-level, Word-level

Dataset: Stanford IMDB sentiment corpus

## Install Env

```shell
conda env create -f environment.yml
conda activate mytorch
```

## Run

1. download dataset 

```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xvf aclImdb_v1.tar.gz -C /data1/
```

2. generate dataset and vocabulary files

``` 
python preprocess.py 
```
output files:
+ data_dir + "/dataset_train.pkl"
+ data_dir + "/dataset_valid.pkl"
+ data_dir + "/dataset_test.pkl"
+ ./conf/word_vocab.pkl

3. train and eval

```
sh run_train.sh
sh run_eval.sh
```

## Results

GPU: P40 (one card)

Model  | Train accuracy | Validation accuracy | Test accuracy | Train epochs | Train time (min)
:------------- | :---: |:---: | :---: | :----: | :----:
char-CNN  | 0.9595 |0.8675 | 0.86264 | 100 | 31.13 |
char-LSTM | 0.98635 |0.62377 | 0.60924 | 100 | 19.91 |
char-Transformer | 0.65377 | 0.64565 | 0.64220 | 100 | 73.48 |
char-DPCNN | 1.00000 | 0.84019 | 0.84352 | 100 | 32.64|
word-CNN  | 0.99995 |0.88215 | 0.86616 | 100 | 7.93 |
word-LSTM | 0.99895 |0.84300 | 0.82328 | 100 | 3.10 |
word-Transformer | 1.00000 | 0.85023 | 0.81120 | 100 | 10.06 |
word-DPCNN | 1.00000 | 0.84642 | 0.84036 | 100 | 8.92|

It seems that:
+ char-CNN and char-DPCNN are good
+ char-Transformer is underfitting
+ char-LSTM is overfitting.
+ all word-leval models are not bad

## Dataset Desc

It's a sentiment corpus, 25000 train samples and 25000 test samples.

```
aclImdb/
|-- README
|-- imdb.vocab
|-- imdbEr.txt
|-- test
|   |-- labeledBow.feat
|   |-- neg
|   |-- pos
|   |-- urls_neg.txt
|   `-- urls_pos.txt
`-- train
    |-- labeledBow.feat
    |-- neg
    |-- pos
    |-- unsup
    |-- unsupBow.feat
    |-- urls_neg.txt
    |-- urls_pos.txt
    `-- urls_unsup.txt
```

## Other

DPCNN paper: https://riejohnson.com/paper/dpcnn-acl17.pdf

DPCNN comment: https://zhuanlan.zhihu.com/p/35457093

RNN tips: https://www.zhihu.com/question/57828011

Conda manual: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Datasets: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

Datasets: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

## TODO

Add pretrained word embeddings

https://nlp.stanford.edu/projects/glove/
