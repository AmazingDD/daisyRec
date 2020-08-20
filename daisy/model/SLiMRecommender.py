#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17
Modified on 21/8/2020
@author: Maurizio Ferrari Dacrema, Yu Di
@Description: Modify this source file and change it to adapt to daisyRec mode, the original author is Maurizio Ferrari Dacrema
"""

import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet

# TODO this recommender must change to multiprocessing mode and compress into a more beautiful way


class SLIM(object):
    def __init__(self, user_num, item_num, topk=100,
                 l1_ratio=0.1, alpha=1.0, positive_only=True):
        """
        SLIM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        topk : int, Top-K number
        l1_ratio : float, The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`
        alpha : float, Constant that multiplies the penalty terms
        positive_only : bool, When set to True, forces the coefficients to be positive
        """
        self.md = ElasticNet(alpha=alpha, 
                             l1_ratio=l1_ratio, 
                             positive=positive_only, 
                             fit_intercept=False,
                             copy_X=False,
                             precompute=True,
                             selection='random',
                             max_iter=100,
                             tol=1e-4)
        self.item_num = item_num
        self.user_num = user_num
        self.topk = topk

        self.w_sparse = None
        self.A_tilde = None

        print(f'user num: {user_num}, item num: {item_num}')

    def fit(self, df, verbose=True):
        train = self._convert_df(self.user_num, self.item_num, df)

        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for currentItem in range(self.item_num):
            y = train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = train.indptr[currentItem]
            end_pos = train.indptr[currentItem + 1]

            current_item_data_backup = train.data[start_pos: end_pos].copy()
            train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.md.fit(train, y)

            nonzero_model_coef_index = self.md.sparse_coef_.indices
            nonzero_model_coef_value = self.md.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topk)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            train.data[start_pos:end_pos] = current_item_data_backup

            if verbose and (time.time() - start_time_printBatch > 300 or (currentItem + 1) % 1000 == 0 or currentItem == self.item_num - 1):
                print('{}: Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}'.format(
                     'SLIMElasticNetRecommender',
                     currentItem+1,
                     100.0* float(currentItem+1)/self.item_num,
                     (time.time()-start_time)/60,
                     float(currentItem)/(time.time()-start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.w_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                      shape=(self.item_num, self.item_num), dtype=np.float32)

        train = train.tocsr()
        self.A_tilde = train.dot(self.w_sparse).A

    def predict(self, u, i):

        return self.A_tilde[u, i]

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make WRMF available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat
