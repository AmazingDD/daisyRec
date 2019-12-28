'''
@Author: Yu Di
@Date: 2019-12-03 14:53:45
@LastEditors  : Yudi
@LastEditTime : 2019-12-28 22:46:49
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from daisy.model import slim

class SLIM(object):
    def __init__(self, user_num, item_num, alpha=0.5, lam_bda=0.02, 
            max_iter=1000, tol=0.0001, lambda_is_ratio=True):
        self.user_num = user_num
        self.item_num = item_num
        
        self.alpha = alpha
        self.lam_bda = lam_bda
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_is_ratio = lambda_is_ratio

        print('Sparse Linear Matrix algorithm...')

        self.alpha = alpha # lasso regularization ratio
        self.lam_bda = lam_bda # elastic net coefficients
        self.max_iter = max_iter  # max learning iteration
        self.tol = tol  # learning threshold
        self.lambda_is_ratio = lambda_is_ratio  # lambda is ratio?

        self.W = None  # coefficients set dimension: item_num * item_num

    def fit(self, data:pd.DataFrame):
        A = np.zeros((self.user_num, self.item_num))
        for _, row in data.iterrows():
            user, item = int(row['user']), int(row['item'])
            A[user, item] = 1
        self.A = A # user-item matrix

        print(f'Start calculate matrix W（alpha={self.alpha}, lambda={self.lam_bda}, max_iter={self.max_iter}, tol={self.tol})')
        self.W = self.__aggregation_coefficients()

        self.A_tilde = self.A.dot(self.W)
        print('Finish generate predction user-item matrix......')

    def predict(self, u, i):
        return self.A_tilde[u, i]

    def __aggregation_coefficients(self):
        group_size = 100  # row/col number for oncurrent calculation
        n = self.item_num // group_size  # group number for concurrent calculation
        starts = []
        ends = []
        for i in range(n):
            start = i * group_size
            starts.append(start)
            ends.append(start + group_size)
        if self.item_num % group_size != 0:
            starts.append(n * group_size)
            ends.append(self.item_num)
            n += 1

        print('covariance updates pre-calculating')
        covariance_array = None
        with ProcessPoolExecutor() as executor:
            covariance_array = np.vstack(list(executor.map(slim.compute_covariance, 
                                                           [self.A] * n, 
                                                           starts, 
                                                           ends)))
        slim.symmetrize_covariance(covariance_array)

        print('coordinate descent for learning W matrix......')
        if self.lambda_is_ratio:
            with ProcessPoolExecutor() as executor:
                return np.hstack(list(executor.map(slim.coordinate_descent_lambda_ratio, 
                                                   [self.alpha] * n, 
                                                   [self.lam_bda] * n, 
                                                   [self.max_iter] * n, 
                                                   [self.tol] * n, 
                                                   [self.user_num] * n, 
                                                   [self.item_num] * n, 
                                                   [covariance_array] * n, 
                                                   starts, 
                                                   ends)))
        else:
            with ProcessPoolExecutor() as executor:
                return np.hstack(list(executor.map(slim.coordinate_descent, 
                                                   [self.alpha] * n, 
                                                   [self.lam_bda] * n, 
                                                   [self.max_iter] * n, 
                                                   [self.tol] * n, 
                                                   [self.user_num] * n, 
                                                   [self.item_num] * n, 
                                                   [covariance_array] * n, 
                                                   starts, 
                                                   ends)))
