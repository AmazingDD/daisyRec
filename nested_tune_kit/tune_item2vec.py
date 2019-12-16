'''
@Author: Yu Di
@Date: 2019-12-04 21:25:49
@LastEditors: Yudi
@LastEditTime: 2019-12-16 22:25:00
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from daisy.model.Item2VecRecommender import Item2Vec, SGNS
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur, BuildCorpus, PermutedSubsampledCorpus

def get_weights(wc, idx2item, ss_t, whether_weights):
    wf = np.array([wc[item] for item in idx2item])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2item)
    weights = wf if whether_weights else None

    return vocab_size, weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Item2Vec recommender test')
    # common settings
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset preprocess op.: origin/5core/10core')
    parser.add_argument('--topk', 
                        type=int, 
                        default=50, 
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='loo', 
                        help='method for split test,options: loo/fo/tfo/tloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=.2, 
                        help='split ratio for test set')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='loo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, 
                        help='No. of candidates item for predict')
    # algo settings
    parser.add_argument('--unk', type=str, default='<UNK>', help='UNK token')
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_item', type=int, default=20000, help="maximum number of item set")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epochs', type=int, default=6, help="number of epochs") # 100
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    args = parser.parse_args()

    # TODO generate algo paramter settings for grid-search tuning

    '''Validation Process for Parameter Tuning'''
    df, user_num, item_num = load_rate(args.dataset, args.prepro)
    train_set, test_set = split_test(df, args.test_method, args.test_size)

    # pre-build Corpus for all item
    pre = BuildCorpus(df, args.window, args.max_item, args.unk)
    pre.build()

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)

    train_set_list, val_set_list, fn = split_validation(train_set, 
                                                        args.val_method, 
                                                        args.fold_num)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    # store metrics result for test set
    fnl_metric = []
    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        train = train_set_list[fold]
        validation = val_set_list[fold]

        # get ground truth
        train_ur = get_ur(train)
        val_ur = get_ur(validation)

        # build recommender model
        dt = pre.convert(train)
        vocab_size, weights = get_weights(pre.wc, pre.idx2item, args.ss_t, args.weights)
        
        embed_model = Item2Vec(item_num=vocab_size, embedding_size=args.e_dim)
        model = SGNS(embedding=embed_model, item_num=vocab_size, n_negs=args.n_negs, weights=weights)

        dataset = PermutedSubsampledCorpus(dt)  
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True) 

        model.fit(dataloader, args.epochs, pre.item2idx)
        model.build_user_vec(train_ur)

        # build candidates set
        assert max([len(v) for v in val_ur.values()]) < candidates_num, 'Small candidates_num setting'
        val_ucands = defaultdict(list)
        for k, v in val_ur.items():
            sample_num = candidates_num - len(v)
            sub_item_pool = item_pool - v - train_ur[k] # remove GT & interacted
            sample_num = min(len(sub_item_pool), sample_num)
            samples = random.sample(sub_item_pool, sample_num)
            val_ucands[k] = list(v | set(samples))

        # get predict result
        print('')
        print('Generate recommend list...')
        print('')
        preds = {}
        for u in tqdm(val_ucands.keys()):
            pred_rates = [model.predict(u, i) for i in val_ucands[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(val_ucands[u])[rec_idx]
            preds[u] = top_n

        # convert rank list to binary-interaction
        for u in preds.keys():
            preds[u] = [1 if i in val_ur[u] else 0 for i in preds[u]]

        # calculate metrics for validation set
        pre_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        rec_k = recall_at_k(preds, val_ur, args.topk)
        hr_k = hr_at_k(preds, val_ur)
        map_k = map_at_k(preds.values())
        mrr_k = mrr_at_k(preds, args.topk)
        ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])

        print('-'*20)
        print(f'Precision@{args.topk}: {pre_k:.4f}')
        print(f'Recall@{args.topk}: {rec_k:.4f}')
        print(f'HR@{args.topk}: {hr_k:.4f}')
        print(f'MAP@{args.topk}: {map_k:.4f}')
        print(f'MRR@{args.topk}: {mrr_k:.4f}')
        print(f'NDCG@{args.topk}: {ndcg_k:.4f}')

        tmp_metric = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])
        fnl_metric.append(tmp_metric)

    # get final validation metrics result by average operation
    fnl_metric = np.array(fnl_metric).mean(axis=0)
    print('='*20, 'Metrics for All Validation', '='*20)
    print(f'Precision@{args.topk}: {fnl_metric[0]:.4f}')
    print(f'Recall@{args.topk}: {fnl_metric[1]:.4f}')
    print(f'HR@{args.topk}: {fnl_metric[2]:.4f}')
    print(f'MAP@{args.topk}: {fnl_metric[3]:.4f}')
    print(f'MRR@{args.topk}: {fnl_metric[4]:.4f}')
    print(f'NDCG@{args.topk}: {fnl_metric[5]:.4f}')

    # record all tuning result and settings