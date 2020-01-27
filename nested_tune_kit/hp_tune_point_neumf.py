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

from daisy.model.pointwise.NeuMFRecommender import PointNeuMF
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur, negative_sampling, PointMFData

def sigmoid(x):
    return 1/(1 + np.exp(-x))

parser = argparse.ArgumentParser(description='Point-Wise MF recommender test')
# tune settings
parser.add_argument('--sc_met', 
                    type=str, 
                    default='ndcg', 
                    help='use which metric to define hyperopt score')
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
                    default=10, 
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
# algo settings
parser.add_argument('--num_ng', 
                    type=int, 
                    default=4, 
                    help='negative sampling number')
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
                    default=50, 
                    help='training epochs')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=256, 
                    help='batch size for training')
parser.add_argument('--lamda', 
                    type=float, 
                    default=0.0, 
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
args = parser.parse_args()

tune_log_path = './tune_log/'
if not os.path.exists(tune_log_path):
    os.makedirs(tune_log_path)

f = open(tune_log_path + f'{args.loss_type}-neumf_{args.dataset}_{args.prepro}_{args.val_method}.csv', 
         'w', 
         encoding='utf-8')
f.write('Pre,Rec,HR,MAP,MRR,NDCG,num_ng,factors,layers,dropout,lr,batch_size,lamda' + '\n')

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

if args.dataset in ['yelp', 'amazon-electronic']:
    space = {
        'num_ng': hp.quniform('num_ng', 1, 5, 1),
        'factor_num': hp.quniform('factor_num', 1, 100, 1),
        'num_layers': hp.quniform('num_layers', 1, 3, 1),
        'dropout': hp.uniform('dropout', 0, 1),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [256, 512, 1024]),
        'lamda': hp.loguniform('lamda', np.log(1e-4), np.log(1e-2))
    }
else:
    space = {
        'num_ng': hp.quniform('num_ng', 1, 5, 1),
        'factor_num': hp.quniform('factor_num', 1, 100, 1),
        'num_layers': hp.quniform('num_layers', 1, 3, 1),
        'dropout': hp.uniform('dropout', 0, 1),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [64, 128, 256, 512]),
        'lamda': hp.loguniform('lamda', np.log(1e-4), np.log(1e-2))
    }

metric_idx = {
    'precision': 0,
    'recall': 1,
    'hr': 2,
    'map': 3,
    'mrr': 4, 
    'ndcg': 5,
}

def opt_func(params, mi=args.sc_met, topk=args.topk):
    num_ng, factor_num, num_layers = int(params['num_ng']), int(params['factor_num']), int(params['num_layers'])
    dropout, lr, batch_size, lamda = params['dropout'], params['lr'], params['batch_size'], params['lamda']
    print(f'Parameter Settings: num_ng:{num_ng},factors:{factor_num},layers:{num_layers},dropout:{dropout},lr:{lr},batch_size:{batch_size},lamda:{lamda}')

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
        train_sampled = negative_sampling(user_num, item_num, train, num_ng)
        # format training data
        train_dataset = PointMFData(train_sampled)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=4)

        # whether load pre-train model
        model_name = args.model_name
        assert model_name in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
        GMF_model_path = f'./tmp/{args.dataset}/CL/GMF.pt'
        MLP_model_path = f'./tmp/{args.dataset}/CL/MLP.pt'
        NeuMF_model_path = f'./tmp/{args.dataset}/CL/NeuMF.pt'

        if model_name == 'NeuMF-pre':
            assert os.path.exists(GMF_model_path), 'lack of GMF model'    
            assert os.path.exists(MLP_model_path), 'lack of MLP model'
            GMF_model = torch.load(GMF_model_path)
            MLP_model = torch.load(MLP_model_path)
        else:
            GMF_model = None
            MLP_model = None

        # build recommender model
        model = PointNeuMF(user_num, item_num, factor_num, num_layers, dropout, 
                            lr, args.epochs, lamda, args.model_name, GMF_model, MLP_model,
                            args.gpu, args.loss_type)
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
            # build a validation MF dataset for certain user u
            tmp = pd.DataFrame({'user': [u for _ in val_ucands[u]], 
                                'item': val_ucands[u], 
                                'rating': [0. for _ in val_ucands[u]], # fake label, make nonsense
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

                prediction = model.predict(user_u, item_i)
                _, indices = torch.topk(prediction, topk)
                top_n = torch.take(torch.tensor(val_ucands[u]), indices).cpu().numpy()

            preds[u] = top_n

        # convert rank list to binary-interaction
        for u in preds.keys():
            preds[u] = [1 if i in val_ur[u] else 0 for i in preds[u]]

        # calculate metrics for validation set
        pre_k = np.mean([precision_at_k(r, topk) for r in preds.values()])
        rec_k = recall_at_k(preds, val_ur, topk)
        hr_k = hr_at_k(preds, val_ur)
        map_k = map_at_k(preds.values())
        mrr_k = mrr_at_k(preds, topk)
        ndcg_k = np.mean([ndcg_at_k(r, topk) for r in preds.values()])

        tmp_metric = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])
        fnl_metric.append(tmp_metric)

    # get final validation metrics result by average operation
    fnl_metric = np.array(fnl_metric).mean(axis=0)
    print('='*20, 'Metrics for All Validation', '='*20)
    print(f'Precision@{topk}: {fnl_metric[0]:.4f}')
    print(f'Recall@{topk}: {fnl_metric[1]:.4f}')
    print(f'HR@{topk}: {fnl_metric[2]:.4f}')
    print(f'MAP@{topk}: {fnl_metric[3]:.4f}')
    print(f'MRR@{topk}: {fnl_metric[4]:.4f}')
    print(f'NDCG@{topk}: {fnl_metric[5]:.4f}')

    score = fnl_metric[metric_idx[mi]]

    # record all tuning result and settings
    fnl_metric = [f'{mt:.4f}' for mt in fnl_metric]
    line = ','.join(fnl_metric) + f',{num_ng},{factor_num},{num_layers},{dropout},{lr},{batch_size},{lamda}' + '\n'

    f.write(line)
    f.flush()

    return -score

if __name__ == '__main__':
    best = fmin(opt_func, space, algo=tpe.suggest, max_evals=30)
    
    f.close()
