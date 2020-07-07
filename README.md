![DaisyRec](pics/logo.png)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-daisy) [![Version](https://img.shields.io/badge/version-v1.1.2-orange)](https://github.com/AmazingDD/daisyRec) ![GitHub repo size](https://img.shields.io/github/repo-size/amazingdd/daisyrec) ![GitHub](https://img.shields.io/github/license/amazingdd/daisyrec)

## Overview

<!-- ![daisyRec's structure](pics/structure.png) -->

DaisyRec is a Python toolkit dealing with rating prediction and item ranking issue.

The name DAISY (roughly :) ) stands for multi-**D**imension f**A**irly compAr**I**son for recommender **SY**stem.

To get all dependencies, run:

    pip install -r requirement.txt

Before running, you need first run: 

    python setup.py build_ext --inplace

to generate `.so` or `.pyd` file used for further import.

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it.

DaisyRec handled ranking issue mainly and split recommendation problem into point-wise ones and pair-wise ones so that different loss function are constructed such as BPR, Top-1, Hinge and Cross Entropy. All algorithms already implemented are exhibited below:

<img src="pics/algos.jpeg" width="40%" height="30%" style="margin: auto; cursor:default" />

use `main.py` to achieve KPI results calculated by certain algorithm above. For example, you can implement this program to implement BPR-MF:

    python main.py --problem_type=pair --algo_name=mf --loss_type=BPR --num_ng=2

## TODO list

- [ ] weight initialization interface
- [x] user-level time-aware fold-out method
- [x] user-level/item-level/user-item-level Ncore

## Cite

Here is a Bibtex entry if you ever need to cite **DaisyRec** in a research paper (please keep us posted, we would love to know if Daisy was helpful to you)

```
@misc{yudi2020daisyrec,
  title={DaisyRec},
  author={Yu Di},
  year={2020},
  publisher={GitHub},
  howpublished={\url{https://github.com/AmazingDD/daisyRec}},
}
```
