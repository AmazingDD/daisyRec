![DaisyRec](pics/logo.png)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-daisy) [![Version](https://img.shields.io/badge/version-v1.1.2-orange)](https://github.com/AmazingDD/daisyRec) ![GitHub repo size](https://img.shields.io/github/repo-size/amazingdd/daisyrec) ![GitHub](https://img.shields.io/github/license/amazingdd/daisyrec)



## Overview

DaisyRec is a Python toolkit dealing with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for Multi-**D**imension f**AI**rly comp**A**r**I**son for recommender **SY**stem. The whole framework of Daisy is showed below:

<img src="pics/DiasyRec.png" align="center" width="75%" style="margin: 0 auto">

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it.

We will consistently update this repo.
<!-- A more professional daisyRec could be checked in `dev` branch -->

## Datasets

You can download experiment data, and put them into the `data` folder.
All data are available in links below: 

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

## How to run

1. Make sure running command `python setup.py build_ext --inplace` to compile dependent extensions before running the other code. After that, you will find file \*.so or \*.pyd file generated in path `daisy/model/`

2. In order to reproduce results, you need run `python data_generator.py` to create `experiment_data` folder with certain public dataset listed in our paper. If you just want to research one certain dataset, you need to modify the code in `data_generator.py` to indicate your demands and let this code yield train and test datasets as you want. In the default situation, `data_generator.py` will generate all kinds of datasets (raw data, 5-core data and 10-core data) with different data splitting methods, including `tloo`, `loo`, `tfo` and `fo`. The meaning of these split methods will be explained in the `Important Commands` of `README`.

<!-- 3. There are parameter tuning codes for validation dataset stored in `nested_tune_kit` and KPI-generating code for test dataset stored in `test_kit`. Each of the code in these folders should be moved into the root path, just the same hierarchy as `data_generator.py`, so that every user could successfully implement this code. Furthermore, if you have an IDE toolkit, you can simply set your work path and run in any folder path. -->

3. There are seperate codes for validation and test, and they are stored in `nested_tune_kit` and `test_kit`, respectively. Each of the code in these folders should be moved into the root path, just the same hierarchy as `data_generator.py` to successfully run these code. Furthermore, if you have an IDE toolkit, you can simply set your work path and run in any folder path. 

<!--4. As we all know, validation dataset is used for parameter tuning, so we provide *split_validation* interface inside all codes in the `nested_tune_kit` folder. Further and more detail parameter settings information about validation split method is depicted in `daisy/utils/loader.py` -->

4. The validation dataset is used for parameter tuning, so we provide *split_validation* interfact inside the code in the `nested_tune_kit` folder. Further and more detail parameter settings information about validation split method is depicted in `daisy/utils/loader.py`. After finishing validation, the results will be stored in the generated file `tune_log/`.

5. Based on the best parameter determined by the validation, run the test code that you moved before and the results will be stored in the generated file `res/`.

<!-- 5. After finished operations above, you can just run the code you moved before and wait for the result files generated in `tune_log/` for the tuning procedure or `res/` for the test set ranking results, which is dynamically created while running.

## Examples to run:

<!-- What should I do with daisyRec if I want to reproduce the top-20 result published like *BPR-MF* with ML-1M-10core dataset(When tuning, we fix sample method as uniform method).-->

Taking the following case as an example: if we want to reproduce the top-20 results for *BPR-MF* on ML-1M-10core dataset.


<!--
1. Assume you have already run `data_generator.py` and get `tfo` (time-aware split by ratio method) test dataset, you must get files named `train_ml-1m_10core_tfo.dat`, `test_ml-1m_10core_tfo.dat` in `./experiment_data/`. **This step is essential!**
-->

1. Assume we have already run `data_generator.py` and get the training and test datasets by `tfo` (i.e., time-aware split by ratio method). We should get files named `train_ml-1m_10core_tfo.dat`, `test_ml-1m_10core_tfo.dat` in `./experiment_data/`.

2. The whole procedure contains tuning and testing. Therefore, we first need run `hp_tune_pair_mf.py` to get the best parameter settings. Command to run:

```
python hp_tune_pair_mf.py --dataset=ml-1m --prepro=10core --val_method=tfo --topk=20 --loss_type=BPR --sample_method=uniform --gpu=0
```
  Since all reasonable parameter search scope was fixed in the code, there is no need to parse more arguments
  
<!--3. After you finished step 2 and just get the best parameter settings from `tune_log/` or you just wanna reproduce the results provided in paper, you can run the following command to achieve it.-->

3. After finishing step 2, we will get the best paramter settings from `tune_log/`. Then we can run the test code by following the command as below:


```
python run_pair_mf.py --dataset=ml-1m --prepro=10core --test_method=tfo --topk=20 --loss_type=BPR --num_ng=2 --factors=34 --epochs=50 --lr=0.0005 --lamda=0.0016 --sample_method=uniform --gpu=0
```

More details of arguments are available in help message, try:

```
python run_pair_mf.py --help
```

4. Once step 3 terminated, we can obtain the results w.r.t. top-20 from the dynamically generated result file `./res/ml-1m/10core_tfo_pairmf_BPR_uniform.csv`

---

## Important Commands

The description of all common parameter settings used by code inside `examples` are listed below:

| Commands      | Description on Commands                                      | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choices&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                                                       | Description on Choices                                       |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| dataset       | the selected datasets                                        | ml-100k;<br>ml-1m;<br>ml-10m;<br>ml-20m;<br>lastfm;<br>bx;<br>amazon-cloth;<br>amazon-electronic;<br>amazon-book;<br>amazon-music;<br>epinions;<br>yelp;<br>citeulike;<br>netflix | all choices are the names of datasets                          |
| prepro        | The data pre-processing methods                              | origin;<br>Ncore                                         | 'origin' means using the raw data; <br>'Ncore' means only preserving users and items that  have interactions more than **N**.  Notice **N** could be any integer value |
| topk          | the length of recommendation list                            |                                                              |                                                              |
| test_method   | method of training-test data splitting                       | fo<br>tfo<br>loo<br>tloo                              | split by ratio<br>time-aware split by ratio<br>leave one  out<br>time-aware leave one out |
| test_size     | ratio of test set size                                       |                                                              |                                                              |
| val_method    | method of training-validation splitting                      | cv<br>fo<br>tfo<br>tloo<br>loo<br>ufo             | Combine with fold_num => fold_num-CV;<br>combine with fold_num & val_size =>fold_num-Split by  ratio(9:1);<br>Split by ratio with timestamp, combine with val_size => 1-Split by  ratio(9:1);<br>Leave one out with timestamp => 1-Leave one out;<br>Combine with fold_num => fold_num-Leave one out;<br>split by ratio in user level with K-fold; |
| fold_num      | the number of fold used for validation(only  work when 'cv', 'fo' is set). |                                                              |                                                              |
| cand_num      | the number of candidate items used for ranking             |                                                              |                                                              |
| sample_method | negative sampling method                                     | uniform<br>item-ascd<br>item-desc                      | uniformly sampling;<br>sampling popular items with low rank;<br>sampling popular item with high rank |
| num_ng        | the number of negative samples                               |                                                              |                                                              |


<!--The other parameters used for specific algorithms are listed in paper.-->

## More Ranking Results

More ranking results for different methods on different datasets across various settings of top-N (N=1,5,10,20,30) are available in the file of `ranking_results.md`.
