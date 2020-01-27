![DaisyRec](logo.png)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-daisy) [![Version](https://img.shields.io/badge/version-v1.1.2-orange)](https://github.com/AmazingDD/daisyRec) ![GitHub repo size](https://img.shields.io/github/repo-size/amazingdd/daisyrec) ![GitHub](https://img.shields.io/github/license/amazingdd/daisyrec)

## Overview

DaisyRec is a Python toolkit dealing with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for Multi-**D**imension f**AI**rly comp**A**r**I**son for recommender **SY**stem.

1. You can also download experiment data from links below: 

    - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
    - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
    - [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
    - [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
    - [Netflix Prize Data](https://archive.org/download/nf_prize_dataset.tar)
    - [Last.fm](https://grouplens.org/datasets/hetrec-2011/)
    - [Book Crossing](https://grouplens.org/datasets/book-crossing/)
    - [Epinions](http://www.cse.msu.edu/~tangjili/trust.html)
    <!-- - [Pinterest](https://sites.google.com/site/xueatalphabeta/academic-projects) -->
    - [CiteULike](https://github.com/js05212/citeulike-a)
    - [Amazon-Book](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv)
    - [Amazon-Electronic](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv)
    - [Amazon-Cloth](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv)
    - [Amazon-Music](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv)
    - [Yelp Challenge](https://kaggle.com/yelp-dataset/yelp-dataset)

    then put certain dataset into corresponding folder in `data` folder.

2. Item-Ranking recommendation algorithms reimplementation with pytorch.

3. To get all dependencies, run:

    `pip install -r requirement.txt`

3. Before running, you need first run: 
`python setup.py build_ext --inplace` 
to generate `.so` file for `macOS` or `.pyd` file for `WindowsOS` used for further import.

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

#### _Point-wise with cross-entropy-loss(CL)/square-loss(SL)_ 

- MF
- FM
- NeuMF

#### _Pair-wise with bpr-loss(BPR)/hinge-loss(HL)_

- MF
- FM
- NeuMF

## Examples to run:

The default top-K number is set to 50, you can change top-K number by modifying `topk` argument in `run_*.py`.

```
python run_itemknn.py --sim_method=pearson --topk=15
python run_point_fm.py --loss_type=CL
```

More details of arguments are available in help message, try:

```
python run_itemknn.py --help
```

---

## Dependencies

- torch (>=1.1.0)
- Numpy (>=1.18.0)
- Pandas (>=0.24.0)
- scikit-learn (>=0.21.3)

## Metrics

- Pre (Precision)
- Rec (Recall)
- HR (Hit Rate)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- F1 (F1-Score)
- AUC (Area Under Curve)

---

<!-- ## Cite

Here is a Bibtex entry if you ever need to cite **Daisy** in a research paper (please keep us posted, we would love to know if Daisy was helpful to you)

```
@Misc{,
author =   {},
title =    {},
howpublished = {\url{}},
year = {2020}
}
``` -->

## Development Status

Starting from version 1.0.0, we will only provide bugfixes.

For bugs, issues or questions about Daisy, please use the GitHub project page.

In the future, we will integrate more algorithms and finish all KPI report.

We expect more contributors join us for building a better **Daisy**.

## Appendix

**Reference**

* SLIM: Sparse Linear Methods for Top-N Recommender Systems
* Probabilistic matrix factorization
* Performance of recommender algorithms on top-N recommendation tasks
* Factorization meets the neighborhood: a multifaceted collaborative filtering model
* Collaborative Filtering for Implicit Feedback Datasets
* BPR: Bayesian Personalized Ranking from Implicit Feedback

* Factorization Machines
* Neural Factorization Machines for Sparse Predictive Analytics
* Neural Collaborative Filtering
* Item2Vec: Neural Item Embedding for Collaborative Filtering
* AutoRec: Autoencoders Meet Collaborative Filtering
<!-- | eALS | EALSRecommender.py | Fast Matrix Factorization for Online Recommendation with Implicit Feedback | -->
