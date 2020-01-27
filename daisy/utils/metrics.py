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
    # # another way for calculating hit rate
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

def auc_at_k(rs):
    uauc = 0.
    for user in rs.keys():
        label_all = rs[user]

        pos_num = len(list(filter(lambda x: x == 1, label_all)))
        neg_num = len(label_all) - pos_num

        pos_rank_num = 0
        for j in range(len(pred_all)):
            if label_all[j] == 1:
                pos_rank_num += j + 1

        auc = (pos_rank_num - pos_num * (pos_num + 1) / 2) / (pos_num * neg_num)

        uauc += auc

    return uauc / len(rs)

def f1_at_k(rs, test_ur):
    uf1 = 0.
    for user in rs.keys():
        r = rs[user]
        r = np.asarray(r) != 0
        # start calculate precision
        prec_k = sum(r) / len(r)
        # start calculate recall
        if len(test_ur[user]) == 0:
            raise KeyError(f'Invalid User Index: {user}')
        rec_k = sum(r) / len(test_ur[user])
        # start calculate f1-score
        f1_k = 2 * prec_k * rec_k / (rec_k + prec_k)

        uf1 += f1_k

    return uf1 / len(rs)
