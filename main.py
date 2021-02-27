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
from daisy.utils.splitter import split_test
from daisy.utils.data import PointData, PairData, UAEData
from daisy.utils.loader import load_rate, get_ur, convert_npy_mat, build_candidates_set
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k


if __name__ == '__main__':
    ''' all parameter part '''
    args = parse_args()

    # store running time in time_log file
    time_log = open('time_log.txt', 'a') 
    
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

    print('='*50, '\n')
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

    if args.algo_name in ['cdae', 'vae']:
        train_dataset = UAEData(user_num, item_num, train_set, test_set)
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
                factors=args.factors,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                gpuid=args.gpu
            )
        elif args.algo_name == 'fm':
            from daisy.model.point.FMRecommender import PointFM
            model = PointFM(
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
            from daisy.model.point.NeuMFRecommender import PointNeuMF
            model = PointNeuMF(
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
            from daisy.model.point.NFMRecommender import PointNFM
            model = PointNFM(
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
        elif args.algo_name == 'cdae':
            from daisy.model.CDAERecommender import CDAE
            model = CDAE(
                rating_mat=training_mat,
                factors=args.factors,
                act_activation=args.act_func,
                out_activation=args.out_func,
                epochs=args.epochs,
                lr=args.lr,
                q=args.dropout,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                loss_type=args.loss_type,
                gpuid=args.gpu
            )
        elif args.algo_name == 'vae':
            from daisy.model.VAERecommender import VAE
            model = VAE(
                rating_mat=training_mat,
                q=args.dropout,
                epochs=args.epochs,
                lr=args.lr,
                reg_1=args.reg_1,
                reg_2=args.reg_2,
                beta=args.kl_reg,
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
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )

    # build recommender model
    s_time = time.time()
    model.fit(train_loader)
    elapsed_time = time.time() - s_time
    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_{args.problem_type}{args.algo_name}_{args.loss_type}_{args.sample_method},{elapsed_time:.4f}' + '\n')
    time_log.close()

    print('Start Calculating Metrics......')
    test_ucands = build_candidates_set(test_ur, total_train_ur, item_pool, candidates_num)

    # get predict result
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    if args.algo_name in ['vae', 'cdae'] and args.problem_type == 'point':
        for u in tqdm(test_ucands.keys()):
            pred_rates = [model.predict(u, i) for i in test_ucands[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_ucands[u])[rec_idx]
            preds[u] = top_n
    else:
        for u in tqdm(test_ucands.keys()):
            # build a test MF dataset for certain user u to accelerate
            tmp = pd.DataFrame({
                'user': [u for _ in test_ucands[u]], 
                'item': test_ucands[u], 
                'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
            })
            tmp_neg_set = sampler.transform(tmp, is_training=False)
            tmp_dataset = PairData(tmp_neg_set, is_training=False)
            tmp_loader = data.DataLoader(
                tmp_dataset,
                batch_size=candidates_num, 
                shuffle=False, 
                num_workers=0
            )
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
                top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

            preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/{args.prepro}/{args.test_method}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    for k in [1, 5, 10, 20, 30, 50]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if k == 10:
            print(f'Precision@{k}: {pre_k:.4f}')
            print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            print(f'MAP@{k}: {map_k:.4f}')
            print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    common_prefix = f'with_{args.sample_ratio}{args.sample_method}'
    algo_prefix = f'{args.loss_type}_{args.problem_type}_{args.algo_name}'

    res.to_csv(
        f'{result_save_path}{algo_prefix}_{common_prefix}_kpi_results.csv', 
        index=False
    )
