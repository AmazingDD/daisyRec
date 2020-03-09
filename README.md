![DaisyRec](logo.png)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-daisy) [![Version](https://img.shields.io/badge/version-v1.1.2-orange)](https://github.com/AmazingDD/daisyRec) ![GitHub repo size](https://img.shields.io/github/repo-size/amazingdd/daisyrec) ![GitHub](https://img.shields.io/github/license/amazingdd/daisyrec)

## Overview

DaisyRec is a Python toolkit dealing with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for Multi-**D**imension f**AI**rly comp**A**r**I**son for recommender **SY**stem.

To get all dependencies, run:

    pip install -r requirement.txt

Before running, you need first run: 

    python setup.py build_ext --inplace

to generate `.so` file for `macOS` or `.pyd` file for `WindowsOS` used for further import.

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it.

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

<!-- ## Appendix

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
* AutoRec: Autoencoders Meet Collaborative Filtering -->
<!-- | eALS | EALSRecommender.py | Fast Matrix Factorization for Online Recommendation with Implicit Feedback | -->
