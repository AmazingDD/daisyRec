import os
import torch
import datetime
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

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

def get_history_matrix(df, config, row='user', use_config_value_name=False):
    '''
    get the history interactions by user/item
    '''
    logger = config['logger']
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

def get_inter_matrix(df, config, form='coo'):
    '''
    get the whole sparse interaction matrix
    '''
    logger = config['logger']
    value_field = config['INTER_NAME']
    src_field, tar_field = config['UID_NAME'], config['IID_NAME']
    user_num, item_num = config['user_num'], config['item_num']

    src, tar = df[src_field].values, df[tar_field].values
    data = df[value_field].values

    mat = sp.coo_matrix((data, (src, tar)), shape=(user_num, item_num))

    if form == 'coo':
        return mat
    elif form == 'csr':
        return mat.tocsr()
    else:
        raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented...')

