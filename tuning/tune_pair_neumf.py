'''
@Author: Yu Di
@Date: 2019-12-05 10:41:50
@LastEditors: Yudi
@LastEditTime: 2019-12-14 00:42:53
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

from daisy.model.pairwise.NeuMFRecommender import PairNeuMF
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur, PairMFData
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pair-Wise NeuMF recommender test')
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
                        default='cv', 
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
    parser.add_argument('--factor_num', 
                        type=int, 
                        default=32, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=3, 
                        help='number of layers in MLP model')
    parser.add_argument('--model_name', 
                        type=str, 
                        default='NeuMF-end', 
                        help='target model name, if NeuMF-pre plz run MLP and GMF before')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.0, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=1024, 
                        help='batch size for training')
    parser.add_argument('--lamda', 
                        type=float, 
                        default=0.0, 
                        help='regularizer weight')
    parser.add_argument('--out', 
                        default=True, 
                        help='save model or not')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    args = parser.parse_args()

    # TODO generate algo paramter settings for grid-search tuning

    '''Validation Process for Parameter Tuning'''
    df, user_num, item_num = load_rate(args.dataset, args.prepro)
    train_set, test_set = split_test(df, args.test_method, args.test_size)

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)

    train_set_list, val_set_list, fn = split_validation(train_set, 
                                                        args.val_method, 
                                                        args.fold_num)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num
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
        train_dataset = PairMFData(train, user_num, item_num, args.num_ng)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                       shuffle=True, num_workers=4)

        # whether load pre-train model
        model_name = args.model_name
        assert model_name in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
        GMF_model_path = f'./tmp/{args.dataset}/BPR/GMF.pt'
        MLP_model_path = f'./tmp/{args.dataset}/BPR/MLP.pt'
        NeuMF_model_path = f'./tmp/{args.dataset}/BPR/NeuMF.pt'

        if model_name == 'NeuMF-pre':
            assert os.path.exists(GMF_model_path), 'lack of GMF model'    
            assert os.path.exists(MLP_model_path), 'lack of MLP model'
            GMF_model = torch.load(GMF_model_path)
            MLP_model = torch.load(MLP_model_path)
        else:
            GMF_model = None
            MLP_model = None

        # build recommender model
        model = PairNeuMF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, 
                          args.lr, args.epochs, args.lamda, args.model_name, 
                          GMF_model, MLP_model, args.gpu, args.loss_type)
        model.fit(train_loader)

        # build candidates set
        assert max([len(v) for v in val_ur.values()]) < candidates_num, 'Small candidates_num setting'
        val_ucands = defaultdict(list)
        for k, v in val_ur.items():
            sample_num = candidates_num - len(v)
            sub_item_pool = item_pool - v - train_ur[k] # remove GT & interacted
            samples = random.sample(sub_item_pool, sample_num)
            val_ucands[k] = list(v | set(samples))

        # get predict result
        print('')
        print('Generate recommend list...')
        print('')
        preds = {}
        for u in tqdm(val_ucands.keys()):
            # build a validation MF dataset for certain user u to accelerate
            tmp = pd.DataFrame({'user': [u for _ in val_ucands[u]], 
                                'item': val_ucands[u], 
                                'rating': [0. for _ in val_ucands[u]], # fake label, make nonsense
                            })
            tmp_dataset = PairMFData(tmp, user_num, item_num, 0, False)
            tmp_loader = data.DataLoader(tmp_dataset, batch_size=candidates_num, 
                                         shuffle=False, num_workers=0)
            # get top-N list with torch method 
            for items in tmp_loader:
                user_u, item_i = items[0], items[1]
                if torch.cuda.is_available():
                    user_u = user_u.cuda()
                    item_i = item_i.cuda()
                else:
                    user_u = user_u.cpu()
                    item_i = item_i.cpu()

                prediction = model.predict(user_u, item_i)
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