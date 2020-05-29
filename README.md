![DaisyRec](logo.png)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-daisy) [![Version](https://img.shields.io/badge/version-v1.1.2-orange)](https://github.com/AmazingDD/daisyRec) ![GitHub repo size](https://img.shields.io/github/repo-size/amazingdd/daisyrec) ![GitHub](https://img.shields.io/github/license/amazingdd/daisyrec)

## Overview

DaisyRec is a Python toolkit dealing with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for Multi-**D**imension f**AI**rly comp**A**r**I**son for recommender **SY**stem.

You can download experiment data from links below: 

  - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
  - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
  - [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
  - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
  - [Netflix Prize Data](https://archive.org/download/nf_prize_dataset.tar)
  - [Last.fm](https://grouplens.org/datasets/hetrec-2011/)
  - [Book Crossing](https://grouplens.org/datasets/book-crossing/)
  - [Epinions](http://www.cse.msu.edu/~tangjili/trust.html)
  - [CiteULike](https://github.com/js05212/citeulike-a)
  - [Amazon-Book](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv)
  - [Amazon-Electronic](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv)
  - [Amazon-Cloth](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv)
  - [Amazon-Music](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv)
  - [Yelp Challenge](https://kaggle.com/yelp-dataset/yelp-dataset)

  then put certain dataset into corresponding folder in `data` folder.

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it.

## List of all algorithms

#### _Common Algo._

- Popular
- Item-KNN
- User-KNN
- WRMF
- PureSVD
- Item2Vec
- BiasMF
- RSVD2
- SVD++
- AutoEncoder
- NFM
- DeepFM

#### _Point-wise with cross-entropy-loss(CL)/square-loss(SL)_ 

- MF
- FM
- NeuMF

#### _Pair-wise with bpr-loss(BPR)/hinge-loss(HL)_

- MF
- FM
- NeuMF

## How to run
this repo is just a temporary project used for submitting, 

1. Make sure running command `python setup.py build_ext --inplace` to compile dependent extensions before running the other code. After that, you will find file \*.so or \*.pyd file generated in path `daisy/model/`

2. In order to reproduce results, you need run `python data_generator.py` to create `experiment_data` folder with certain public dataset listed in our paper. If you just wanna research one certain dataset, you need modify code in `data_generator.py` to indicate and let this code yield train dataset and test dataset as you wish.

3. There are parameter tuning code for validation dataset stored in `nested_tune_kit` and KPI-generating code for test dataset stored in `examples`. Each of the code in these folders should be moved in the root path, just the same hierarchy as `data_generator.py`, so that every user could successfully implement.

4. As we all know, validation dataset is used for parameter tuning, so we provide *split_validation* interface inside all codes in `nested_tune_kit` folder. Further and more detail parameter setting information about validation split method is depicted in `daisy/utils/loader.py`

5. After finished operations above, you can just run the code you moved before and wait for the results generated in `tune_log/` or `res/` folder, which was dynamically created while running.

## Examples to run:

What should I do with daisyRec if I want to reproduce the top-20 result published like *BPR-MF* with ML-1M-10core dataset(When tuning, we fix sample method as uniform method).

1. Assume you have already run `data_generator.py` and get TFO(time-aware split by ratio method) test dataset, you must get files named `train_ml-1m_10core_tfo.dat`, `test_ml-1m_10core_tfo.dat` in `./experiment_data/`. **This step is essential!**

2. The whole procedure contains tuning and testing. Therefore, we need run `hp_tune_pair_mf.py` to get the best parameter settings. Command to run:
```
python hp_tune_pair_mf.py --dataset=ml-1m --prepro=10core --val_method=tfo --test_method=tfo --topk=20 --loss_type=BPR --sample_method=uniform --gpu=0
  ```
  Since all reasonable parameter search scope was fixed in the code, there is no need to parse more arguments
  
3. After you finished step 2 and just get the best parameter settings from `tune_log/` or you just wanna reproduce the results provided in paper, you can run the following command to achieve it.

```
python run_pair_mf.py --dataset=ml-1m --prepro=10core --val_method=tfo --test_method=tfo --topk=20 --loss_type=BPR --num_ng=2 --factors=34 --epochs=50 --lr=0.0005 --lamda=0.0016 --sample_method=uniform --gpu=0
```

More details of arguments are available in help message, try:

```
python run_pair_mf.py --help
```

4. After terminated step 3, you can take the results from the dynamically generated result file `./res/ml-1m/10core_tfo_pairmf_BPR_uniform.csv`

---

## Parameter settings

The description of all common parameter settings used by code inside `examples` are listed below:

 - dataset 
  
    dataset used for experiments, 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'bx', 'amazon-cloth','amazon-electronic', 'amazon-book', 'amazon-music', 'epinions', 'yelp', 'citeulike','netflix'

  - prepro

    method to processing the raw data, 'origin' for raw data, 'Ncore' for preserving user and item both have interactions more than **N**

  - topk

    the length of rank list

  - test_method

    method of train test split. 
    'fo': split by ratio
    'tfo': split by ratio with timesstamp
    'tloo': leave one out with timestamp
    'loo': leave one out
    'ufo': split by ratio in user level

  - test_size
    ratio of test set size

  - val_method

    method of train validation split
    'cv': combine with fold_num => fold_num-CV
    'fo': combine with fold_num & val_size => fold_num-Split by ratio(9:1)
    'tfo': Split by ratio with timestamp, combine with val_size => 1-Split by ratio(9:1)
    'tloo': Leave one out with timestamp => 1-Leave one out
    'loo': combine with fold_num => fold_num-Leave one out
    'ufo': split by ratio in user level with K-fold

  - fold_num

    the number of fold used for validation. only work when 'cv', 'fo' is set

  - cand_num

    the number of candidated items used for ranking

  - sample_method

    negative sampling method, default is 'uniform' , 'item-ascd' for popular item with low rank, 'item-desc' for popular item with high rank

  - num_ng

    the number of negative samples

  the other parameters used for specific algorithms are listed in paper.
   

