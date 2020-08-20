import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data as data

from daisy.utils.sampler import Sampler
from daisy.utils.parser import parse_args
from daisy.utils.data import PointData, PairData, UAEData
from daisy.utils.splitter import split_test, split_validation
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k

def opt_func(user_num, item_num, train_set, args):
    # retrain model by the whole train set
    # format training data
    sampler = Sampler(
        user_num, 
        item_num, 
        num_ng=args.num_ng, 
        sample_method=args.sample_method, 
        sample_ratio=args.sample_ratio
    )
    neg_set = sampler.transform(train_set, is_training=True)


if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    ''' Test Process for Metrics Exporting '''
    df, user_num, item_num = load_rate(args.dataset, args.prepro, binary=False)
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    if args.dataset in ['yelp']:
        train_set['timestamp'] = pd.to_datetime(train_set['timestamp'])
        test_set['timestamp'] = pd.to_datetime(test_set['timestamp'])
    df = pd.concat([train_set, test_set], ignore_index=True)
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    train_set['rating'] = 1.0
    test_set['rating'] = 1.0

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    train_set_list, val_set_list, fn = split_validation(train_set, 
                                                    args.val_method, 
                                                    args.fold_num)

    print('='*50, '\n')
    # store metrics result for final validation set
    fnl_metric = []
    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        train = train_set_list[fold]
        validation = val_set_list[fold]

        # get ground truth
        train_ur = get_ur(train)
        val_ur = get_ur(validation)

        # start negative sampling
        sampler = Sampler(
            user_num, 
            item_num, 
            num_ng=args.num_ng, 
            sample_method=args.sample_method, 
            sample_ratio=args.sample_ratio
        )
        neg_set = sampler.transform(train_set, is_training=True)

    