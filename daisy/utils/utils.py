import os
import torch
import logging
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

from .metrics import Metric
from .config import metrics_name_config

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

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
    ensure_dir(path)

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

def get_ur(df):
    """
    Method of getting user-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur

def get_ir(df):
    """
    Method of getting item-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir

def build_candidates_set(test_ur, train_ur, config, drop_past_inter=True):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_num : No. of all items
    cand_num : int, the number of candidates
    drop_past_inter : drop items already appeared in train set

    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    item_num = config['item_num']
    candidates_num = config['cand_num']

    test_ucands, test_u = [], []
    for u, r in test_ur.items():
        sample_num = candidates_num - len(r) if len(r) <= candidates_num else 0
        if sample_num == 0:
            samples = np.random.choice(list(r), candidates_num)
        else:
            pos_items = list(r) + list(train_ur[u]) if drop_past_inter else list(r)
            neg_items = np.setdiff1d(np.arange(item_num), pos_items)
            samples = np.random.choice(neg_items, size=sample_num)
            samples = np.concatenate((samples, list(r)), axis=None)

        test_ucands.append([u, samples])
        test_u.append(u)
    
    return test_u, test_ucands

def get_adj_mat(n_users, n_items):
    """
    method of get Adjacency matrix
    Parameters
    --------
    n_users : int, the number of users
    n_items : int, the number of items

    Returns
    -------
    adj_mat: adjacency matrix
    norm_adj_mat: normal adjacency matrix
    mean_adj_mat: mean adjacency matrix
    """
    logger = logging.getLogger()
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    logger.info('already create adjacency matrix', adj_mat.shape)

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        logger.info('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)

        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        logger.info('check normalized adjacency matrix whether equal to this laplacian matrix.')
        return temp

    norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = mean_adj_single(adj_mat)

    logger.info('already normalize adjacency matrix')
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

def get_history_matrix(df, config, row='user', use_config_value_name=False):
    logger = logging.getLogger()
    assert row in df.columns, f'invalid name {row}: not in columns of history dataframe'
    uid_name, iid_name  = config['UID_NAME'], config['IID_NAME']
    user_ids, item_ids = df[uid_name].values, df[iid_name].values
    value_name = config['INTER_NAME'] if use_config_value_name else None

    user_num, item_num = config['user_num'], config['item_num']
    values = np.ones(len(df)) if value_name is None else df[value_name].values

    if row == 'user':
        row_num, max_col_num = user_num, item_num
        row_ids, col_ids = user_ids, item_ids
    else: # 'item'
        row_num, max_col_num = item_num, user_num
        row_ids, col_ids = item_ids, user_ids

    history_len = np.zeros(row_num, dtype=np.int64)
    for row_id in row_ids:
        history_len[row_id] += 1

    col_num = np.max(history_len)
    if col_num > max_col_num * 0.2:
        logger.info(f'Max value of {row}\'s history interaction records has reached: {col_num / max_col_num * 100:.4f}% of the total.')

    history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
    history_value = np.zeros((row_num, col_num))
    history_len[:] = 0
    for row_id, value, col_id in zip(row_ids, values, col_ids):
        history_matrix[row_id, history_len[row_id]] = col_id
        history_value[row_id, history_len[row_id]] = value
        history_len[row_id] += 1

    return torch.LongTensor(history_matrix), torch.FloatTensor(history_value), torch.LongTensor(history_len)
