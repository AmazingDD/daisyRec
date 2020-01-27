# import numpy as np
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor

# from daisy.model import slim

# class SLIM(object):
#     def __init__(self, user_num, item_num, alpha=0.5, lam_bda=0.02, 
#             max_iter=1000, tol=0.0001, lambda_is_ratio=True):
#         self.user_num = user_num
#         self.item_num = item_num
        
#         self.alpha = alpha
#         self.lam_bda = lam_bda
#         self.max_iter = max_iter
#         self.tol = tol
#         self.lambda_is_ratio = lambda_is_ratio

#         print('Sparse Linear Matrix algorithm...')

#         self.alpha = alpha # lasso regularization ratio
#         self.lam_bda = lam_bda # elastic net coefficients
#         self.max_iter = max_iter  # max learning iteration
#         self.tol = tol  # learning threshold
#         self.lambda_is_ratio = lambda_is_ratio  # lambda is ratio?

#         self.W = None  # coefficients set dimension: item_num * item_num

#     def fit(self, data:pd.DataFrame):
#         A = np.zeros((self.user_num, self.item_num))
#         for _, row in data.iterrows():
#             user, item = int(row['user']), int(row['item'])
#             A[user, item] = 1
#         self.A = A # user-item matrix

#         print(f'Start calculate matrix W（alpha={self.alpha}, lambda={self.lam_bda}, max_iter={self.max_iter}, tol={self.tol})')
#         self.W = self.__aggregation_coefficients()

#         self.A_tilde = self.A.dot(self.W)
#         print('Finish generate predction user-item matrix......')

#     def predict(self, u, i):
#         return self.A_tilde[u, i]

#     def __aggregation_coefficients(self):
#         group_size = 100  # row/col number for oncurrent calculation
#         n = self.item_num // group_size  # group number for concurrent calculation
#         starts = []
#         ends = []
#         for i in range(n):
#             start = i * group_size
#             starts.append(start)
#             ends.append(start + group_size)
#         if self.item_num % group_size != 0:
#             starts.append(n * group_size)
#             ends.append(self.item_num)
#             n += 1

#         print('covariance updates pre-calculating')
#         covariance_array = None
#         with ProcessPoolExecutor() as executor:
#             covariance_array = np.vstack(list(executor.map(slim.compute_covariance, 
#                                                            [self.A] * n, 
#                                                            starts, 
#                                                            ends)))
#         slim.symmetrize_covariance(covariance_array)

#         print('coordinate descent for learning W matrix......')
#         if self.lambda_is_ratio:
#             with ProcessPoolExecutor() as executor:
#                 return np.hstack(list(executor.map(slim.coordinate_descent_lambda_ratio, 
#                                                    [self.alpha] * n, 
#                                                    [self.lam_bda] * n, 
#                                                    [self.max_iter] * n, 
#                                                    [self.tol] * n, 
#                                                    [self.user_num] * n, 
#                                                    [self.item_num] * n, 
#                                                    [covariance_array] * n, 
#                                                    starts, 
#                                                    ends)))
#         else:
#             with ProcessPoolExecutor() as executor:
#                 return np.hstack(list(executor.map(slim.coordinate_descent, 
#                                                    [self.alpha] * n, 
#                                                    [self.lam_bda] * n, 
#                                                    [self.max_iter] * n, 
#                                                    [self.tol] * n, 
#                                                    [self.user_num] * n, 
#                                                    [self.item_num] * n, 
#                                                    [covariance_array] * n, 
#                                                    starts, 
#                                                    ends)))
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet

class SLIM(object):
    def __init__(self, user_num, item_num, topk=100,
                 l1_ratio=0.1, alpha=1.0, positive_only=True):
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
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(self.item_num, self.item_num), dtype=np.float32)

        train = train.tocsr()
        self.A_tilde = train.dot(self.W_sparse).A

    def predict(self, u, i):

        return self.A_tilde[u, i]

    def _convert_df(self, user_num, item_num, df):
        '''Process Data to make WRMF available'''
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat

