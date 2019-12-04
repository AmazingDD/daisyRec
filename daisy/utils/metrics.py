'''
@Author: Yu Di
@Date: 2019-12-02 22:40:23
@LastEditors: Yudi
@LastEditTime: 2019-12-04 14:29:08
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: metrics for evaluating recommend list
'''
import numpy as np

def precision_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    '''
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    # return np.mean(r)
    return sum(r) / len(r)

def recall_at_k(rs, test_ur, k):
    assert k >= 1
    res = []
    for user in test_ur.keys():
        r = np.asarray(rs[user])[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        if len(test_ur[user]) == 0:
            raise KeyError(f'Invalid User Index: {user}')
        res.append(sum(r) / len(test_ur[user]))

    return np.mean(res)

def mrr_at_k(rs, k):
    assert k >= 1
    res = 0
    for r in rs.values():
        r = np.asarray(r)[:k] != 0 
        for index, item in enumerate(r):
            if item == 1:
                res += 1 / (index + 1)
    return res / len(rs)

def ap(r):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    '''
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / len(r)

def map_at_k(rs):
    '''
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    '''
    return np.mean([ap(r) for r in rs])

def dcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    '''
    assert k >= 1
    r = np.asfarray(r)[:k] != 0
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Normalized discounted cumulative gain
    '''
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def hr_at_k(rs, test_ur):
    # TODO Do-check
    # numer, denom = 0., 0.
    # for user in test_ur.keys():
    #     numer += np.sum(rs[user])
    #     denom += len(test_ur[user])

    # return numer / denom
    uhr = 0
    for r in rs.values():
        if np.sum(r) != 0:
            uhr += 1
    
    return uhr / len(rs)
