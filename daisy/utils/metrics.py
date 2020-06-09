import numpy as np


def precision_at_k(r, k):
    """
    Precision calculation method
    Parameters
    ----------
    r : List, list of the rank items
    k : int, top-K number

    Returns
    -------
    pre : float, precision value
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    # return np.mean(r)
    pre = sum(r) / len(r)

    return pre


def recall_at_k(rs, test_ur, k):
    """
    Recall calculation method
    Parameters
    ----------
    rs : Dict, {user : rank items} for test set
    test_ur : Dict, {user : items} for test set ground truth
    k : int, top-K number

    Returns
    -------
    rec : float recall value
    """
    assert k >= 1
    res = []
    for user in test_ur.keys():
        r = np.asarray(rs[user])[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        if len(test_ur[user]) == 0:
            raise KeyError(f'Invalid User Index: {user}')
        res.append(sum(r) / len(test_ur[user]))
    rec = np.mean(res)

    return rec


def mrr_at_k(rs, k):
    """
    Mean Reciprocal Rank calculation method
    Parameters
    ----------
    rs : Dict, {user : rank items} for test set
    k : int, topK number

    Returns
    -------
    mrr : float, MRR value
    """
    assert k >= 1
    res = 0
    for r in rs.values():
        r = np.asarray(r)[:k] != 0 
        for index, item in enumerate(r):
            if item == 1:
                res += 1 / (index + 1)
    mrr = res / len(rs)

    return mrr


def ap(r):
    """
    Average precision calculation method
    Parameters
    ----------
    r : List, Relevance scores (list or numpy) in rank order (first element is the first item)

    Returns
    -------
    a_p : float, Average precision value
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    a_p = np.sum(out) / len(r)

    return a_p


def map_at_k(rs):
    """
    Mean Average Precision calculation method
    Parameters
    ----------
    rs : Dict, {user : rank items} for test set

    Returns
    -------
    m_a_p : float, MAP value
    """
    m_a_p = np.mean([ap(r) for r in rs])
    return m_a_p


def dcg_at_k(r, k):
    """
    Discounted Cumulative Gain calculation method
    Parameters
    ----------
    r : List, Relevance scores (list or numpy) in rank order
                (first element is the first item)
    k : int, top-K number

    Returns
    -------
    dcg : float, DCG value
    """
    assert k >= 1
    r = np.asfarray(r)[:k] != 0
    if r.size:
        dcg = np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
        return dcg
    return 0.


def ndcg_at_k(r, k):
    """
    Normalized Discounted Cumulative Gain calculation method
    Parameters
    ----------
    r : List, Relevance scores (list or numpy) in rank order
            (first element is the first item)
    k : int, top-K number

    Returns
    -------
    ndcg : float, NDCG value
    """
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    ndcg = dcg_at_k(r, k) / idcg

    return ndcg


def hr_at_k(rs, test_ur):
    """
    Hit Ratio calculation method
    Parameters
    ----------
    rs : Dict, {user : rank items} for test set
    test_ur : (Deprecated) Dict, {user : items} for test set ground truth

    Returns
    -------
    hr : float, HR value
    """
    # another way for calculating hit rate
    # numer, denom = 0., 0.
    # for user in test_ur.keys():
    #     numer += np.sum(rs[user])
    #     denom += len(test_ur[user])

    # return numer / denom
    uhr = 0
    for r in rs.values():
        if np.sum(r) != 0:
            uhr += 1
    hr = uhr / len(rs)

    return hr


def auc_at_k(rs):
    """
    Area Under Curve calculation method
    Parameters
    ----------
    rs : Dict, {user : rank items} for test set

    Returns
    -------
    m_auc : float, AUC value
    """
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
    m_auc = uauc / len(rs)

    return m_auc


def f1_at_k(rs, test_ur):
    """
    F1-score calculation method
    Parameters
    ----------
    rs : Dict, {user : rank items} for test set
    test_ur : Dict, {user : items} for test set ground truth

    Returns
    -------
    fs : float, F1-score value
    """
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
    fs = uf1 / len(rs)

    return fs
