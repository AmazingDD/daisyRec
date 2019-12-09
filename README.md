# daisyRec

## Overview

DaisyRec is a Python toolkit that deal with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for Multi-**D**imension f**AI**r comp**A**r**I**son for recommender **SY**stem.

1. you can also download experiment data from links below: 

    - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
    - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
    - [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
    - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
    - [Netflix Prize Data](https://archive.org/download/nf_prize_dataset.tar)
    - [Last.fm](https://grouplens.org/datasets/hetrec-2011/)
    - [Book Crossing](https://grouplens.org/datasets/book-crossing/)
    - [Epinions](http://www.cse.msu.edu/~tangjili/trust.html)
    - [Pinterest](https://sites.google.com/site/xueatalphabeta/academic-projects)
    - [CiteULike](https://github.com/js05212/citeulike-a)
    - [Amazon-Book](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv)
    - [Amazon-Electronic](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv)
    - [Amazon-Cloth](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv)
    - [Amazon-Music](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv)
    - [Yelp Challenge](https://kaggle.com/yelp-dataset/yelp-dataset)

    then put certain dataset into corresponding folder in `data` folder.

2. Item-Ranking recommendation algorithms reimplementation with pytorch and tensorflow.

3. To get all dependencies, run:

    `pip install -r requirement.txt`

3. Before running, you need first run: 
`python setup.py build_ext --inplace` 
to generate `.so` file for `macOS` or `.pyd` file for `WindowsOS` used for further import.

make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it.

## List of all algorithms

| Algo. | File |
| ------ | ------ |
| Popular | run_mostpop.py |
| Item-KNN | run_itemknn.py |
| User-KNN | run_userknn.py |
| WRMF | run_wrmf.py |
| PureSVD | run_puresvd.py |
| Item2Vec | run_item2vec.py|
| CL-MF | run_clmf.py |
| BPR-MF | run_bprmf.py |
| HL-MF | run_hlmf.py |
| CL-FM | run_clfm.py |
| BPR-FM | run_bprfm.py |
| HL-FM | run_hlfm.py |
| CL-NeuMF | run_clneumf.py |
| BPR-NeuMF | run_bprneumf.py |
| HL-NeuMF | run_hlneumf.py |
| SLiM (*TODO*) | run_slim.py |

## Examples to run:

Default set top-K number to 10, you can change top-K number by modifying `topk` argument.

```
python run_itemknn.py --sim_method=pearson --topk=15
```

More details of arguments are available in help message, try:

```
python run_itemknn.py --help
```

---

## Benchmarks

Here are the 6 metrics: 

- Pre(Precision)
- Rec(Recall)
- HR(Hit Rate)
- MAP(Mean Average Precision)
- MRR(Mean Reciprocal Rank)
- NDCG(Normalized Discounted Cumulative Gain)

obtained by various algorithms with the best parameter settings. 

The results above are reproducible.

<!-- ## Simple Result Achieved for quick look

| Algo | HR@10 | NDCG@10 | MAP@10 |
| ------ | ------ | ------ | -- |
| Pop | 0.101  | 0.338 | 0.040 |
| UserKNN | 0.141  | 0.341 | 0.069 |
| ItemKNN | 0.153  | 0.351 | 0.079 |
| SLiM | 0.359 | 0.706 | 0.262 |
| NMF | 0.157 | 0.353 | 0.078 |
| PureSVD | 0.347 | 0.638 | 0.248 |
| SVD | 0.164 | 0.365 | 0.087 |
| SVD++ | 0.152 | 0.360 | 0.077 |
| WRMF | 0.586 | 0.833 | 0.451 |
| BPR-MF | 0.705 | 0.407 | 0.315 |
| NeuMF | 0.698  | 0.401 | 0.310 |
| FM | 0.209 | 0.451 | 0.119 | -->
<!-- | NeuFM(deprecated) | 0.214  | 0.453 | 0.119 | -->

---

## License

Here is a Bibtex entry if you ever need to cite **Daisy** in a research paper (please keep us posted, we would love to know if Daisy was helpful to you):

```
@Misc{,
author =   {},
title =    {},
howpublished = {\url{}},
year = {2020}
}
```

## Development Status

Starting from version 1.0.0, we will only provide bugfixes. 

For bugs, issues or questions about Daisy, please use the GitHub project page. 

We expect more contributors join us for building a better **Daisy**.
