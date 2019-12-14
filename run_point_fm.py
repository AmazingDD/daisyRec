'''
@Author: Yu Di
@Date: 2019-12-05 15:35:52
@LastEditors: Yudi
@LastEditTime: 2019-12-14 10:00:58
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

from daisy.model.pointwise.FMRecommender import PointFM
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur
from daisy.utils.loader import build_feat_idx_dict, negative_sampling, PointFMData


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point-Wise FM recommender test')
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
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=4, 
                        help='negative sampling number')
    parser.add_argument('--batch_norm', 
                        default=True, 
                        help='use batch_norm or not')
    parser.add_argument('--dropout',
                        default='[0.5, 0.2]', 
                        help='dropout rate for FM and MLP')
    parser.add_argument('--hidden_factor', 
                        type=int, 
                        default=64, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='batch size for training')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='training epochs')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.05, 
                        help='learning rate')
    parser.add_argument('--lamda', 
                        type=float, 
                        default=0.0, 
                        help='regularizer for bilinear layers')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='CL', 
                        help='loss function type')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    args = parser.parse_args()

    '''Test Process for Metrics Exporting'''
    # state column name for certain data type
    cat_cols=['user', 'item']
    num_cols=[]

    df, user_num, item_num = load_rate(args.dataset, args.prepro)
    train_set, test_set = split_test(df, args.test_method, args.test_size)

    # convert features to mapping dictionary
    feat_idx_dict, num_features = build_feat_idx_dict(df, cat_cols, num_cols)

    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    print('='*50, '\n')
    # retrain model by the whole train set
    # start negative sampling
    train_sampled = negative_sampling(train_set, args.num_ng, 0.)

    # format training data
    train_dataset = PointFMData(train_sampled, feat_idx_dict, cat_cols, num_cols)
    print('Finish construct FM torch-dataset......')
    train_loader = data.DataLoader(train_dataset, drop_last=True, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=4)

    # build recommender model
    model = PointFM(num_features, args.hidden_factor, args.batch_norm, eval(args.dropout), 
                    args.epochs, args.lr, args.lamda, args.gpu, args.loss_type)
    model.fit(train_loader)

    print('Start Calculating Metrics......')
    # build candidates set
    assert max([len(v) for v in test_ur.values()]) < candidates_num, 'Small candidates_num setting'

    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v)
        sub_item_pool = item_pool - v - total_train_ur[k] # remove GT & interacted
        samples = random.sample(sub_item_pool, sample_num)
        test_ucands[k] = list(v | set(samples))

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    for u in tqdm(test_ucands.keys()):
        # build a test FM dataset for certain user u
        tmp = pd.DataFrame({'user': [u for _ in test_ucands[u]], 
                            'item': test_ucands[u], 
                            'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
                            })
        tmp_dataset = PointFMData(tmp, feat_idx_dict, cat_cols, num_cols)
        tmp_loader = data.DataLoader(tmp_dataset, batch_size=candidates_num, 
                                     shuffle=False, num_workers=0)
        # get top-N list with torch method 
        for features, feature_values, _ in tmp_loader:
            if torch.cuda.is_available():
                features = features.cuda()
                feature_values = feature_values.cuda()
            else:
                features = features.cpu()
                feature_values = feature_values.cpu()

            prediction = model.predict(features, feature_values)
            prediction = prediction.clamp(min=-1.0, max=1.0)
            _, indices = torch.topk(prediction, args.topk)
            top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

        preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # calculate metrics for test set
    pre_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
    rec_k = recall_at_k(preds, test_ur, args.topk)
    hr_k = hr_at_k(preds, test_ur)
    map_k = map_at_k(preds.values())
    mrr_k = mrr_at_k(preds, args.topk)
    ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])

    print(f'Precision@{args.topk}: {pre_k:.4f}')
    print(f'Recall@{args.topk}: {rec_k:.4f}')
    print(f'HR@{args.topk}: {hr_k:.4f}')
    print(f'MAP@{args.topk}: {map_k:.4f}')
    print(f'MRR@{args.topk}: {mrr_k:.4f}')
    print(f'NDCG@{args.topk}: {ndcg_k:.4f}')
    print('='* 20, ' Done ', '='*20)

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})

    for k in [1, 5, 10, 20, 30, 50]:
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    res.to_csv(f'{result_save_path}metric_result_pointfm_{args.loss_type}.csv', index=False)
