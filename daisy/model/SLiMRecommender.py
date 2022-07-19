'''
@inproceedings{ning2011slim,
  title={Slim: Sparse linear methods for top-n recommender systems},
  author={Ning, Xia and Karypis, George},
  booktitle={2011 IEEE 11th international conference on data mining},
  pages={497--506},
  year={2011},
  organization={IEEE}
}
@inproceedings{ferrari2019we,
  title={Are we really making much progress? A worrying analysis of recent neural recommendation approaches},
  author={Ferrari Dacrema, Maurizio and Cremonesi, Paolo and Jannach, Dietmar},
  booktitle={Proceedings of the 13th ACM conference on recommender systems},
  pages={101--109},
  year={2019}
}
'''
import sys
import time
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet

from daisy.model.AbstractRecommender import GeneralRecommender

class SLiM(GeneralRecommender):
    def __init__(self, config):
        """
        SLIM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        topk : int, Top-K number, this is used for improving speed
        elastic : float, The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`
        alpha : float, Constant that multiplies the penalty terms
        positive_only : bool, When set to True, forces the coefficients to be positive
        """
        self.md = ElasticNet(alpha=config['alpha'], 
                             l1_ratio=config['elastic'], 
                             positive=True, 
                             fit_intercept=False,
                             copy_X=False,
                             precompute=True,
                             selection='random',
                             max_iter=100,
                             tol=1e-4)
        self.item_num = config['item_num']
        self.user_num = config['user_num']
        self.topk = config['topk']

        self.w_sparse = None
        self.A_tilde = None

        self.logger.info(f'user num: {self.user_num}, item num: {self.item_num}')

    def fit(self, train_set, verbose=True):
        train = self._convert_df(self.user_num, self.item_num, train_set)

        data_block = 10000000

        rows = np.zeros(data_block, dtype=np.int32)
        cols = np.zeros(data_block, dtype=np.int32)
        values = np.zeros(data_block, dtype=np.float32)

        num_cells = 0

        start_time = time.time()
        start_time_print_batch = start_time

        for current_item in range(self.item_num):
            y = train[:, current_item].toarray()

            # set the j-th column of X to zero
            start_pos = train.indptr[current_item]
            end_pos = train.indptr[current_item + 1]

            current_item_data_backup = train.data[start_pos: end_pos].copy()
            train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.md.fit(train, y)

            nonzero_model_coef_index = self.md.sparse_coef_.indices
            nonzero_model_coef_value = self.md.sparse_coef_.data

            # local_topk = min(len(nonzero_model_coef_value) - 1, self.topk)
            # just keep all nonzero coef value for ranking, if you want improve speed, use code above
            local_topk = len(nonzero_model_coef_value) - 1

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topk)[0:local_topk]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):
                if num_cells == len(rows):
                    rows = np.concatenate((rows, np.zeros(data_block, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(data_block, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(data_block, dtype=np.float32)))

                rows[num_cells] = nonzero_model_coef_index[ranking[index]]
                cols[num_cells] = current_item
                values[num_cells] = nonzero_model_coef_value[ranking[index]]

                num_cells += 1

            train.data[start_pos:end_pos] = current_item_data_backup

            if verbose and (time.time() - start_time_print_batch > 300 or (current_item + 1) % 1000 == 0 or current_item == self.item_num - 1):
                self.logger.info(f'SLIM-ElasticNet-Recommender: Processed {current_item + 1} ( {100.0 * float(current_item + 1) / self.item_num:.2f}% ) in {(time.time() - start_time) / 60:.2f} minutes. Items per second: {float(current_item) / (time.time() - start_time):.0f}')

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

        # generate the sparse weight matrix
        self.w_sparse = sp.csr_matrix((values[:num_cells], (rows[:num_cells], cols[:num_cells])),
                                      shape=(self.item_num, self.item_num), dtype=np.float32)

        train = train.tocsr()
        self.A_tilde = train.dot(self.w_sparse).tolil()

    def predict(self, u, i):
        return self.A_tilde[u, i]

    def rank(self, test_loader):
        rec_ids = np.array([])

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy()
            scores = self.A_tilde[us, cands_ids].A
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[:, rank_ids]

            rec_ids = np.vstack([rec_ids, rank_list])

        return rec_ids

    def full_rank(self, u):
        scores = self.A_tilde[u, :].A.squeeze()

        return np.argsort(-scores)[:self.topk]

    def _convert_df(self, user_num, item_num, df):
        """
        Process Data to make WRMF available
        """
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])
        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat
