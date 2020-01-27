import numpy as np
import pandas as pd

class MostPop(object):
    def __init__(self, N=400):
        self.N = N

    def fit(self, train_set):
        '''most popular item'''
        res = train_set['item'].value_counts()
        # self.top_n = res[:self.N].index.tolist()
        self.rank_list = res.index.tolist()[:self.N]

    def predict(self, test_ur, train_ur, topk=10):
        res = {}
        for user in test_ur.keys():
            candidates = self.rank_list
            candidates = [item for item in candidates if item not in train_ur[user]]
            if len(candidates) < topk:
                raise Exception(f'parameter N is too small to get {topk} recommend items')
            res[user] = candidates[:topk]

        return res
