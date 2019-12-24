'''
@Author: Yu Di
@Date: 2019-12-07 00:59:27
@LastEditors  : Yudi
@LastEditTime : 2019-12-24 09:52:01
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
import torch.utils.data as data

from daisy.model.pairwise.FMRecommender import PairFM
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur
from daisy.utils.loader import build_feat_idx_dict, PairFMData
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pair-Wise FM recommender test')
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
                        help='negative sampling number')
    parser.add_argument('--batch_norm', 
                        default=True, 
                        help='use batch_norm or not')
    parser.add_argument('--dropout',
                        default='[0., 0.2]', 
                        help='dropout rate for FM and MLP')
    parser.add_argument('--hidden_factor', 
                        type=int, 
                        default=64, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=2048, 
                        help='batch size for training')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=40, 
                        help='training epochs')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001,  # can be to large
                        help='learning rate')
    parser.add_argument('--lamda', 
                        type=float, 
                        default=0.001, 
                        help='regularizer for bilinear layers')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    args = parser.parse_args()

    # TODO generate algo paramter settings for grid-search tuning
    param_list = []
    for p1 in range(1, 11):
        for p2 in [10, 20, 50, 100, 150, 200]:
            for p3 in [1e-5, 1e-4, 1e-3]:
                for p4 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    param_list.append((p1, p2, p3, p4))

    tune_log_path = './tune_log/'
    if not os.path.exists(tune_log_path):
        os.makedirs(tune_log_path)
    
    f = open(tune_log_path + f'{args.loss_type}-fm_{args.dataset}_{args.prepro}_{args.val_method}.csv', 
             'w', 
             encoding='utf-8')
    f.write('Pre,Rec,HR,MAP,MRR,NDCG,num_ng,hidden_factor,lr,lamda' + '\n')


    '''Validation Process for Parameter Tuning'''
    # state column name for certain data type
    cat_cols=['user', 'item']
    num_cols=[]

    # df, user_num, item_num = load_rate(args.dataset, args.prepro)
    # train_set, test_set = split_test(df, args.test_method, args.test_size)

    # temporary used for tuning test result
    train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    train_set['rating'] = 1.0
    test_set['rating'] = 1.0
    df = pd.concat([train_set, test_set], ignore_index=True)
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    # convert features to mapping dictionary
    feat_idx_dict, num_features = build_feat_idx_dict(df, cat_cols, num_cols)

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)

    train_set_list, val_set_list, fn = split_validation(train_set, 
                                                        args.val_method, 
                                                        args.fold_num)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    for params in param_list:
        num_ng, hidden_factor, lr, lamda = params
        print(f'Parameter Settings: num_ng:{num_ng}, hidden_factor:{hidden_factor}, lr:{lr}, lamda:{lamda}')

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
            train_dataset = PairFMData(train, feat_idx_dict, item_num, num_ng, True)
            print('Finish construct FM torch-dataset......')
            train_loader = data.DataLoader(train_dataset, drop_last=True, batch_size=args.batch_size, 
                                        shuffle=True, num_workers=4)

            # build recommender model
            model = PairFM(num_features, hidden_factor, args.batch_norm, eval(args.dropout), 
                        args.epochs, lr, lamda, args.gpu, args.loss_type)
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
                # build a validation FM dataset for certain user u
                tmp = pd.DataFrame({'user': [u for _ in val_ucands[u]], 
                                    'item': val_ucands[u], 
                                    'rating': [0. for _ in val_ucands[u]], # fake label, make nonsense
                                    })
                tmp_dataset = PairFMData(tmp, feat_idx_dict, item_num, 0, False)
                tmp_loader = data.DataLoader(tmp_dataset, batch_size=candidates_num, 
                                            shuffle=False, num_workers=0)
                # get top-N list with torch method 
                for feat_i, feat_val_i, feat_j, feat_val_j, _ in tmp_loader:
                    if torch.cuda.is_available():
                        feat_i = feat_i.cuda()
                        feat_val_i = feat_val_i.cuda()
                    else:
                        feat_i = feat_i.cpu()
                        feat_val_i = feat_val_i.cpu()
                    
                    prediction = model.predict(feat_i, feat_val_i)
                    prediction = prediction.clamp(min=-1.0, max=1.0)
                    _, indices = torch.topk(prediction, args.topk)
                    top_n = torch.take(torch.tensor(val_ucands[u]), indices).cpu().numpy()

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
        fnl_metric = [f'{mt:.4f}' for mt in fnl_metric]
        line = ','.join(fnl_metric) + f',{num_ng},{hidden_factor},{lr},{lamda}' + '\n'

        f.write(line)
        f.flush()

    f.close()
