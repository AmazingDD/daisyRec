import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from hyperopt import hp, tpe, fmin

import torch
import torch.utils.data as data

from daisy.utils.opt_toolkit import *
from daisy.utils.parser import parse_args
from daisy.utils.splitter import split_test, split_validation
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k

metric_idx = {
    'precision': 0,
    'recall': 1,
    'hr': 2,
    'map': 3,
    'mrr': 4, 
    'ndcg': 5,
}

def opt_func(space):
    args = parse_args()
    mi=args.score_metric
    topk=args.topk

    # specify certain parameter according to algo_name
    factors = int(space['factors']) if 'factors' in space.keys() else args.factors
    lr = space['lr'] if 'lr' in space.keys() else args.lr
    reg_1 = space['reg_1'] if 'reg_1' in space.keys() else args.reg_1
    reg_2 = space['reg_2'] if 'reg_2' in space.keys() else args.reg_2
    num_layers = int(space['num_layers']) if 'num_layers' in space.keys() else args.num_layers
    dropout = space['dropout'] if 'dropout' in space.keys() else args.dropout
    # num_ng is a special paramter, not be used together with those above
    num_ng = int(space['num_ng']) if 'num_ng' in space.keys() else args.num_ng
    batch_size = int(space['batch_size']) if 'batch_size' in space.keys() else args.batch_size

    # declare a list to store metric score in order to use as target for optimization
    fnl_metric = []
    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        train = train_set_list[fold]
        validation = val_set_list[fold]

        train_ur = get_ur(train)
        val_ur = get_ur(validation)

        # start negative sampling, it will automatically check whether you need to sample
        sampler = Sampler(
            user_num, 
            item_num, 
            num_ng=num_ng, 
            sample_method=args.sample_method, 
            sample_ratio=args.sample_ratio
        )
        neg_set = sampler.transform(train, is_training=True)

        # reformat to adapt certain algorithm
        if args.algo_name in ['cdae', 'vae']:
            train_dataset = UAEData(user_num, item_num, train, validation)
            training_mat = convert_npy_mat(user_num, item_num, train_set)
        else:
            if args.problem_type == 'pair':
                train_dataset = PairData(neg_set, is_training=True)
            else:
                train_dataset = PointData(neg_set, is_training=True)

        if args.problem_type == 'point':
            if args.algo_name == 'mf':
                from daisy.model.point.MFRecommender import PointMF
                model = PointMF(
                    user_num, 
                    item_num, 
                    factors=factors,
                    epochs=args.epochs,
                    lr=lr,
                    reg_1=reg_1,
                    reg_2=reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'fm':
                from daisy.model.point.FMRecommender import PointFM
                model = PointFM(
                    user_num, 
                    item_num,
                    factors=factors,
                    epochs=args.epochs,
                    lr=lr,
                    reg_1=reg_1,
                    reg_2=reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'neumf':
                from daisy.model.point.NeuMFRecommender import PointNeuMF
                model = PointNeuMF(
                    user_num, 
                    item_num,
                    factors=factors,
                    num_layers=num_layers,
                    q=dropout,
                    lr=lr,
                    epochs=args.epochs,
                    reg_1=reg_1,
                    reg_2=reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'nfm':
                from daisy.model.point.NFMRecommender import PointNFM
                model = PointNFM(
                    user_num,
                    item_num,
                    factors=args.factors,
                    act_function=args.act_func,
                    num_layers=num_layers,
                    batch_norm=args.no_batch_norm,
                    q=dropout,
                    epochs=args.epochs,
                    lr=lr,
                    reg_1=reg_1,
                    reg_2=reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'cdae':
                from daisy.model.CDAERecommender import CDAE
                model = CDAE(
                    rating_mat=training_mat,
                    factors=factors,
                    act_activation=args.act_func,
                    out_activation=args.out_func,
                    epochs=args.epochs,
                    lr=lr,
                    q=dropout,
                    reg_1=reg_1,
                    reg_2=reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'vae':
                from daisy.model.VAERecommender import VAE
                model = VAE(
                    rating_mat=training_mat,
                    q=dropout,
                    epochs=args.epochs,
                    lr=lr,
                    reg_1=reg_1,
                    reg_2=reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            else:
                raise ValueError('Invalid algorithm name')
        elif args.problem_type == 'pair':
            if args.algo_name == 'mf':
                from daisy.model.pair.MFRecommender import PairMF
                model = PairMF(
                    user_num, 
                    item_num,
                    factors=args.factors,
                    epochs=args.epochs,
                    lr=args.lr,
                    reg_1=args.reg_1,
                    reg_2=args.reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'fm':
                from daisy.model.pair.FMRecommender import PairFM
                model = PairFM(
                    user_num, 
                    item_num,
                    factors=args.factors,
                    epochs=args.epochs,
                    lr=args.lr,
                    reg_1=args.reg_1,
                    reg_2=args.reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'neumf':
                from daisy.model.pair.NeuMFRecommender import PairNeuMF
                model = PairNeuMF(
                    user_num, 
                    item_num,
                    factors=args.factors,
                    num_layers=args.num_layers,
                    q=args.dropout,
                    lr=args.lr,
                    epochs=args.epochs,
                    reg_1=args.reg_1,
                    reg_2=args.reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            elif args.algo_name == 'nfm':
                from daisy.model.pair.NFMRecommender import PairNFM
                model = PairNFM(
                    user_num, 
                    item_num,
                    factors=args.factors,
                    act_function=args.act_func,
                    num_layers=args.num_layers,
                    batch_norm=args.no_batch_norm,
                    q=args.dropout,
                    epochs=args.epochs,
                    lr=args.lr,
                    reg_1=args.reg_1,
                    reg_2=args.reg_2,
                    loss_type=args.loss_type,
                    gpuid=args.gpu
                )
            else:
                raise ValueError('Invalid algorithm name')
        else:
            raise ValueError('Invalid problem type')

        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )

        model.fit(train_loader)
        print('Start Calculating Metrics......')
        test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num)


    

if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    ''' Test Process for Metrics Exporting '''
    df, user_num, item_num = load_rate(args.dataset, args.prepro, binary=True)
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    # temporary used for tuning test result
    # train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    # test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    if args.dataset in ['yelp']:
        train_set['timestamp'] = pd.to_datetime(train_set['timestamp'])
        test_set['timestamp'] = pd.to_datetime(test_set['timestamp'])
    # df = pd.concat([train_set, test_set], ignore_index=True)
    # user_num = df['user'].nunique()
    # item_num = df['item'].nunique()

    # train_set['rating'] = 1.0
    # test_set['rating'] = 1.0

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    train_set_list, val_set_list, fn = split_validation(
        train_set, 
        args.val_method, 
        args.fold_num
    )

    print('='*50, '\n')
    # TODO begin tuning here
    param_limit = param_extract(args)
    param_dict = confirm_space(param_limit)

    space = dict()
    for key, val in param_dict.items():
        if val[3] == 'int':
            space[key] = hp.quniform(key, val[0], val[1], int(val[2]))
        elif val[3] == 'float':
            space[key] = hp.loguniform(key, np.log(val[0]), np.log(val[1]))

    best = fmin(opt_func, space, algo=tpe.suggest, max_evals=args.tune_epochs)



