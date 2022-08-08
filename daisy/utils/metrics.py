import os
import numpy as np
import pandas as pd

metrics_name_config = {
    "recall": 'Recall',
    "mrr": 'MRR',
    "ndcg": 'NDCG',
    "hit": 'Hit Ratio',
    "precision": 'Precision',
    "f1": 'F1-score',
    "auc": 'AUC',
    "coverage": 'Coverage',
    "diversity": 'Diversity',
    "popularity": 'Average Popularity',
}

def calc_ranking_results(test_ur, pred_ur, test_u, config):
    '''
    calculate metrics with prediction results and candidates sets

    Parameters
    ----------
    test_ur : defaultdict(set)
        groud truths for user in test set
    pred_ur : np.array
        rank list for user in test set
    test_u : list
        the user in order from test set
    '''    
    logger = config['logger']
    path = config['res_path']
    if not os.path.exists(path):
        os.makedirs(path)

    metric = Metric(config)
    res = pd.DataFrame({
        'KPI@K': [metrics_name_config[kpi_name] for kpi_name in config['metrics']]
    })

    common_ks = [1, 5, 10, 20, 30, 50]
    if config['topk'] not in common_ks:
        common_ks.append(config['topk'])
    for topk in common_ks:
        if topk > config['topk']:
            continue
        else:
            rank_list = pred_ur[:, :topk]
            kpis = metric.run(test_ur, rank_list, test_u)
            if topk == 10:
                for kpi_name, kpi_res in zip(config['metrics'], kpis):
                    kpi_name = metrics_name_config[kpi_name]
                    logger.info(f'{kpi_name}@{topk}: {kpi_res:.4f}')

            res[topk] = np.array(kpis)

    return res

class Metric(object):
    def __init__(self, config) -> None:
        self.metrics = config['metrics']
        self.item_num = config['item_num']
        self.item_pop = config['item_pop'] if 'coverage' in self.metrics else None
        self.i_categories = config['i_categories'] if 'diversity' in self.metrics else None

    def run(self, test_ur, pred_ur, test_u):
        res = []
        for mc in self.metrics:
            if mc == "coverage":
                kpi = Coverage(pred_ur, self.item_num)
            elif mc == "popularity":
                kpi = Popularity(test_ur, pred_ur, test_u, self.item_pop)
            elif mc == "diversity":
                kpi = Diversity(pred_ur, self.i_categories)
            elif mc == 'ndcg':
                kpi = NDCG(test_ur, pred_ur, test_u)
            elif mc == 'mrr':
                kpi = MRR(test_ur, pred_ur, test_u)
            elif mc == 'recall':
                kpi = Recall(test_ur, pred_ur, test_u)
            elif mc == 'precision':
                kpi = Precision(test_ur, pred_ur, test_u)
            elif mc == 'hit':
                kpi = HR(test_ur, pred_ur, test_u)
            elif mc == 'map':
                kpi = MAP(test_ur, pred_ur, test_u)
            elif kpi == 'f1':
                kpi = F1(test_ur, pred_ur, test_u)
            elif kpi == 'auc':
                kpi = AUC(test_ur, pred_ur, test_u)
            else:
                raise ValueError(f'Invalid metric name {mc}')

            res.append(kpi)
    
        return res

def Coverage(pred_ur, item_num):
    '''
    Ge, Mouzhi, Carla Delgado-Battenfeld, and Dietmar Jannach. "Beyond accuracy: evaluating recommender systems by coverage and serendipity." Proceedings of the fourth ACM conference on Recommender systems. 2010.
    '''
    return len(np.unique(pred_ur)) / item_num

def Popularity(test_ur, pred_ur, test_u, item_pop):
    '''
    Abdollahpouri, Himan, et al. "The unfairness of popularity bias in recommendation." arXiv preprint arXiv:1907.13286 (2019).

    \frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}
    '''
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        i = np.intersect1d(pred, list(gt))
        if len(i):
            avg_pop = np.sum(item_pop[i]) / len(gt)
            res.append(avg_pop)
        else:
            res.append(0)

    return np.mean(res)

def Diversity(pred_ur, i_categories):
    '''
    Intra-list similarity for diversity

    Parameters
    ----------
    pred_ur : np.array
        rank list for each user in test set
    i_categories : np.array
        (item_num, category_num) with 0/1 value
    ''' 
    res = []
    for u in range(len(pred_ur)):
        ILD = []
        for i in range(len(pred_ur[u])):
            item_i_cats = i_categories[pred_ur[u, i]]
            for j in range(i + 1, len(pred_ur[u])):
                item_j_cats = i_categories[pred_ur[u, j]]
                distance = np.linalg.norm(item_i_cats - item_j_cats)
                ILD.append(distance)
        res.append(np.mean(ILD))

    return np.mean(res)

def Precision(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        pre = np.in1d(pred, list(gt)).sum() / len(pred)

        res.append(pre)

    return np.mean(res)

def Recall(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        rec = np.in1d(pred, list(gt)).sum() / len(gt)

        res.append(rec)

    return np.mean(res)

def MRR(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        mrr = 0.
        for index, item in enumerate(pred):
            if item in gt:
                mrr = 1 / (index + 1)
                break
        
        res.append(mrr)

    return np.mean(res)

def MAP(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        r = np.in1d(pred, list(gt))
        out = [r[:k+1].sum() / (k + 1) for k in range(r.size) if r[k]]
        if not out:
            res.append(0.)
        else:
            ap = np.mean(out)
            res.append(ap)

    return np.mean(res)

def NDCG(test_ur, pred_ur, test_u):
    def DCG(r):
        r = np.asfarray(r) != 0
        if r.size:
            dcg = np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
            return dcg
        return 0.

    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        r = np.in1d(pred, list(gt))

        idcg = DCG(sorted(r, reverse=True))
        if not idcg:
            ndcg = 0.
        else:
            ndcg = DCG(r) / idcg

        res.append(ndcg)

    return np.mean(res)

def HR(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        res.append(1 if r.sum() else 0)

    return np.mean(res)

def AUC(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pos_num = r.sum()
        neg_num = len(pred) - pos_num

        pos_rank_num = 0
        for j in range(len(r) - 1):
            if r[j]:
                pos_rank_num += np.sum(~r[j + 1:])

        auc = pos_rank_num / (pos_num * neg_num)
        res.append(auc)
                
    return np.mean(res)

def F1(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pre = r.sum() / len(pred)
        rec = r.sum() / len(gt)

        f1 = 2 * pre * rec / (pre + rec)
        res.append(f1)

    return np.mean(res)
