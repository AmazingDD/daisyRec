'''
@Author: Yu Di
@Date: 2019-12-10 18:49:52
@LastEditors  : Yudi
@LastEditTime : 2019-12-29 14:41:56
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
from hyperopt import hp, tpe, fmin

import torch
import torch.utils.data as data

from daisy.model.pairwise.SLiMRecommender import PairSLiM
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur, PairMFData
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

def sigmoid(x):
    return 1/(1 + np.exp(-x))

parser = argparse.ArgumentParser(description='Pair-Wise SLiM recommender test')
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
                    default='fo', 
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
parser.add_argument('--loss_type', 
                    type=str, 
                    default='BPR', 
                    help='loss function type')
parser.add_argument('--num_ng', 
                    type=int, 
                    default=4, 
                    help='sample negative items for training')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=2048, 
                    help='batch size for training')
parser.add_argument('--epochs', 
                    type=int, 
                    default=40, 
                    help='The number of iteration of the SGD procedure')
parser.add_argument('--lr', 
                    type=float, 
                    default=0.01, 
                    help='learning rate')    
parser.add_argument('--beta', 
                    type=float, 
                    default=0.0, 
                    help='Frobinious regularization')
parser.add_argument('--lamda', 
                    type=float, 
                    default=0.0, 
                    help='lasso regularization')
parser.add_argument('--gpu', 
                    type=str, 
                    default='0', 
                    help='gpu card ID')
args = parser.parse_args()

tune_log_path = './tune_log/'
if not os.path.exists(tune_log_path):
    os.makedirs(tune_log_path)

f = open(tune_log_path + f'{args.loss_type}-slim_{args.dataset}_{args.prepro}_{args.val_method}.csv', 
            'w', 
            encoding='utf-8')
f.write('Pre,Rec,HR,MAP,MRR,NDCG,num_ng,lr,beta,lamda' + '\n')

'''Validation Process for Parameter Tuning'''
# df, user_num, item_num = load_rate(args.dataset, args.prepro)
# train_set, test_set = split_test(df, args.test_method, args.test_size)

# temporary used for tuning test result
train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
if args.dataset in ['yelp']:
    train_set['timestamp'] = pd.to_datetime(train_set['timestamp'])
    test_set['timestamp'] = pd.to_datetime(test_set['timestamp'])

train_set['rating'] = 1.0
test_set['rating'] = 1.0
df = pd.concat([train_set, test_set], ignore_index=True)
user_num = df['user'].nunique()
item_num = df['item'].nunique()

# get ground truth
test_ur = get_ur(test_set)
total_train_ur = get_ur(train_set)

train_set_list, val_set_list, fn = split_validation(train_set, 
                                                    args.val_method, 
                                                    args.fold_num)

# initial candidate item pool
item_pool = set(range(item_num))
candidates_num = args.cand_num

space = {
    'num_ng': hp.choice('num_ng', [1,2,3,4,5,6,7,8,9,10]),
    'lr': hp.choice('lr', [1e-5, 1e-4, 1e-3, 1e-2]),
    'beta': hp.choice('beta', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    'lamda': hp.choice('lamda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
}

def opt_func(params):
    num_ng, lr, beta, lamda = params['num_ng'], params['lr'], params['beta'], params['lamda']
    print(f'Parameter Settings: num_ng:{num_ng}, lr:{lr}, beta:{beta}, lamda:{lamda}')

    # store metrics result for final validation set
    fnl_metric = []
    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        train = train_set_list[fold]
        validation = val_set_list[fold]

        # get ground truth
        train_ur = get_ur(train)
        val_ur = get_ur(validation)

        # format training data
        train_dataset = PairMFData(train, user_num, item_num, num_ng)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=4)
        # build recommender model
        model = PairSLiM(train, user_num, item_num, args.epochs, 
                        lr, beta, lamda, args.gpu, args.loss_type)
        model.fit(train_loader)

        # build candidates set
        val_ucands = defaultdict(list)
        for k, v in val_ur.items():
            sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
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

    score = np.mean(sigmoid(fnl_metric))

    # record all tuning result and settings
    fnl_metric = [f'{mt:.4f}' for mt in fnl_metric]
    line = ','.join(fnl_metric) + f',{num_ng},{lr},{beta},{lamda}' + '\n'

    f.write(line)
    f.flush()

    return -score

if __name__ == '__main__':
    best = fmin(opt_func, space, algo=tpe.suggest, max_evals=30)
    print(best)

    f.close()
