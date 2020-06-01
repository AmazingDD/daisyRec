import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from point_ncf_model import NCF
from ncf_evaluate import point_metrics

from daisy.utils.loader import load_rate, split_test, get_ur, negative_sampling, PointMFData
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k


class NCFData(data.Dataset):
    def __init__(self, features):
        self.features = features
        self.labels = [0 for _ in range(len(features))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features
        labels = self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]

        return user, item ,label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point-Wise NeuMF recommender test')
    # common settings
    parser.add_argument('--dataset', 
                        type=str, 
                        default='lastfm', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='10core', 
                        help='dataset preprocess op.: origin/5core/10core')
    parser.add_argument('--topk', 
                        type=int, 
                        default=50, 
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='tfo', 
                        help='method for split test,options: loo/fo/tfo/tloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=.2, 
                        help='split ratio for test set')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tfo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, 
                        help='No. of candidates item for predict')
    parser.add_argument('--sample_method', 
                        type=str, 
                        default='uniform', 
                        help='negative sampling method, options: uniform, item-ascd, item-desc')
    # algo settings
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=2, 
                        help='negative sampling number')
    parser.add_argument('--factor_num', 
                        type=int, 
                        default=32, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=2, 
                        help='number of layers in MLP model')
    parser.add_argument('--model_name', 
                        type=str, 
                        default='NeuMF-end', 
                        help='target model name, if NeuMF-pre plz run MLP and GMF before')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.560823, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0035872, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=512, 
                        help='batch size for training')
    parser.add_argument('--lamda', 
                        type=float, 
                        default=0.00113932, 
                        help='regularizer weight')
    parser.add_argument('--out', 
                        default=True, 
                        help='save model or not')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='CL', 
                        help='loss function type')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--loss_back', 
                        type=str, 
                        default='sum', 
                        help='back forward method')
    args = parser.parse_args()

    # store running time
    time_log = open('time_log.txt', 'a')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    '''Test Process for Metrics Exporting'''
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

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    print('='*50, '\n')
    # retrain model by the whole train set
    # start negative sampling
    train_sampled = negative_sampling(user_num, item_num, train_set, 
                                      args.num_ng, sample_method=args.sample_method)
    # format training data
    train_dataset = PointMFData(train_sampled)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=4)

    # whether load pre-train model
    model_name = args.model_name
    assert model_name in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
    GMF_model_path = f'./tmp/{args.dataset}/{args.loss_type}/GMF.pt'
    MLP_model_path = f'./tmp/{args.dataset}/{args.loss_type}/MLP.pt'
    NeuMF_model_path = f'./tmp/{args.dataset}/{args.loss_type}/NeuMF.pt'

    if model_name == 'NeuMF-pre':
        assert os.path.exists(GMF_model_path), 'lack of GMF model'    
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    # build recommender model
    model = NCF(user_num, item_num, args.factor_num, args.num_layers, 
                args.dropout, args.model_name, GMF_model, MLP_model)
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    loss_function = nn.BCEWithLogitsLoss(reduction=args.loss_back)

    if args.model_name == 'NeuMF-pre':
	    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
	    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Start Calculating Metrics......')
    # build candidates set
    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        sub_item_pool = item_pool - v - total_train_ur[k] # remove GT & interacted
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))
        assert len(test_ucands[k]) == args.cand_num, print(len(test_ucands[k]))

    s_time = time.time()
    count, best_ndcg = 0, 0
    for epoch in range(args.epochs):
        model.train() # Enable dropout (if have).
        start_time = time.time()

        for user, item, label in tqdm(train_loader):
            if torch.cuda.is_available():
                user = user.cuda()
                item = item.cuda()
                label = label.float().cuda()
            else:
                user = user.cpu()
                item = item.cpu()
                label = label.float().cpu()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss += args.lamda * (model.embed_item_GMF.weight.norm() + model.embed_user_GMF.weight.norm())
            loss += args.lamda * (model.embed_item_MLP.weight.norm() + model.embed_user_MLP.weight.norm())
            
            loss.backward()
            optimizer.step()
            count += 1
        
        model.eval()

        # TODO test_loader
        tmp_data = []
        for u in test_ucands.keys():
            for i in test_ucands[u]:
                tmp_data.append([u, i])
        test_dataset = NCFData(tmp_data)
        test_loader = data.DataLoader(test_dataset, batch_size=args.cand_num, shuffle=False, num_workers=0)

        NDCG = point_metrics(model, test_loader, args.topk, test_ur)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("NDCG: {:.3f}".format(np.mean(NDCG)))

        if np.mean(NDCG) > best_ndcg:
            best_ndcg, best_epoch = np.mean(NDCG), epoch
            # best_model = model
            torch.save(model, f'./tmp/best_{args.dataset}_{args.prepro}_{args.test_method}_{args.loss_type}_{args.sample_method}.pt')

    print("End. Best epoch {:03d}: NDCG = {:.3f}".format(best_epoch, best_ndcg))

    elapsed_time = time.time() - s_time
    time_log.write(f'{args.dataset}_{args.prepro}_{args.test_method}_pointneumf_{args.loss_type}_{args.sample_method},{elapsed_time:.4f}' + '\n')
    time_log.close()

    # get predict result with best model
    best_model = torch.load(f'./tmp/best_{args.dataset}_{args.prepro}_{args.test_method}_{args.loss_type}_{args.sample_method}.pt')
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}
    for u in tqdm(test_ucands.keys()):
        # build a test MF dataset for certain user u
        tmp = pd.DataFrame({'user': [u for _ in test_ucands[u]], 
                            'item': test_ucands[u], 
                            'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
                            })
        tmp_dataset = PointMFData(tmp)
        tmp_loader = data.DataLoader(tmp_dataset, batch_size=candidates_num, 
                                        shuffle=False, num_workers=0)

        # get top-N list with torch method 
        for user_u, item_i, _ in tmp_loader:
            if torch.cuda.is_available():
                user_u = user_u.cuda()
                item_i = item_i.cuda()
            else:
                user_u = user_u.cpu()
                item_i = item_i.cpu()

            prediction = best_model.predict(user_u, item_i)
            _, indices = torch.topk(prediction, args.topk)
            top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

        preds[u] = top_n

    # convert rank list to binary-interaction
    for u in preds.keys():
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]

    # whether save pre-trained model if necessary
    if args.out:
        if not os.path.exists(f'./tmp/{args.dataset}/{args.loss_type}/'):
            os.makedirs(f'./tmp/{args.dataset}/{args.loss_type}/')
        torch.save(model, f'./tmp/{args.dataset}/{args.loss_type}/{args.model_name.split("-")[0]}.pt')

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/'
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

    res.to_csv(f'{result_save_path}{args.prepro}_{args.test_method}_pointneumf_{args.loss_type}_{args.sample_method}.csv', 
               index=False)
    print('='* 20, ' Done ', '='*20)
