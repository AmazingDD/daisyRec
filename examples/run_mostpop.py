'''
@Author: Yu Di
@Date: 2019-12-02 21:52:18
@LastEditors: Yudi
@LastEditTime: 2019-12-13 17:44:17
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
import logging
import logging.config
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from daisy.model.MostPopRecommender import MostPop
from daisy.utils.loader import load_rate, split_test, get_ur
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

if __name__ == '__main__':
    # load logger configuration
    logging.config.fileConfig("./log.conf")
    logger_name = 'MostPop'
    logger = logging.getLogger(logger_name)
    logger.debug('MostPop experiment running...')

    parser = argparse.ArgumentParser(description='Most-Popular recommender test')
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
                        default=1000, help='No. of candidates item for predict')
    # algo settings
    parser.add_argument('--pop_n', 
                        type=int, 
                        default=400, 
                        help='Initial selected number of Most-popular')
    args = parser.parse_args()

    '''Test Process for Metrics Exporting'''
    df, user_num, item_num = load_rate(args.dataset, args.prepro)
    train_set, test_set = split_test(df, args.test_method, args.test_size)
    
    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    
    print('='*50, '\n')
    print('Start Calculating Metrics......')
    # get predict result
    # retrain model by the whole train set
    # build recommender model
    model = MostPop(args.pop_n)
    model.fit(train_set)
    preds = model.predict(test_ur, total_train_ur, args.topk)

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

    res.to_csv(f'{result_save_path}metric_result_pop.csv', index=False)
