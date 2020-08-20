#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17
Modified on 21/8/2020
@author: Maurizio Ferrari Dacrema, Yu Di
@Description: Modify this source file and change it to adapt to daisyRec mode, the original author is Maurizio Ferrari Dacrema
"""

import os
import heapq
import numpy as np
import scipy.sparse as sp
from enum import Enum
from collections import defaultdict

from daisy.model.extensions.simlib_python import Compute_Similarity_Python


class SimilarityFunction(Enum):
    COSINE = "cosine"
    PEARSON = "pearson"
    JACCARD = "jaccard"


class Compute_Similarity:
    def __init__(self, dataMatrix, 
                 use_implementation = "density", 
                 similarity = None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """
        self.dense = False

        if similarity is not None:
            args["similarity"] = similarity

        if use_implementation == "density":

            if isinstance(dataMatrix, np.ndarray):
                self.dense = True

            elif isinstance(dataMatrix, sp.spmatrix):
                shape = dataMatrix.shape

                num_cells = shape[0]*shape[1]

                sparsity = dataMatrix.nnz/num_cells

                self.dense = sparsity > 0.5

            else:
                print("Compute_Similarity: matrix type not recognized, calling default...")
                use_implementation = "python"

            if self.dense:
                print("Compute_Similarity: detected dense matrix")
                use_implementation = "python"
            else:
                use_implementation = "cython"

        if use_implementation == "cython":
            try:
                from daisy.model.extensions.simlib_cython import Compute_Similarity_Cython
                self.compute_similarity_object = Compute_Similarity_Cython(dataMatrix, **args)

            except ImportError:
                print("Unable to load Cython Compute_Similarity, reverting to Python")
                self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

        elif use_implementation == "python":
            self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

        else:
            raise  ValueError("Compute_Similarity: value for argument 'use_implementation' not recognized")

    def compute_similarity(self,  **args):
        return self.compute_similarity_object.compute_similarity(**args)


class ItemKNNCF(object):
    def __init__(self, user_num, item_num, maxk=40, shrink=100, 
                 similarity='cosine', normalize=True):
        """
        ItemKNN recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        maxk : int, the max similar items number
        shrink : float, shrink similarity value
        similarity : str, way to calculate similarity
        normalize : bool, whether calculate similarity with normalized value
        """
        self.user_num = user_num
        self.item_num = item_num

        self.k = maxk
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity

        self.RECOMMENDER_NAME = "ItemKNNCFRecommender"

        self.pred_mat = None
        self.yr = None

    def fit(self, train_set):
        self.yr = defaultdict(list)
        for _, row in train_set.iterrows():
            self.yr[int(row['user'])].append((int(row['item']), row['rating']))

        train = self._convert_df(self.user_num, self.item_num, train_set)
        
        cold_items_mask = np.ediff1d(train.tocsc().indptr) == 0

        if cold_items_mask.any():
            print("{}: Detected {} ({:.2f} %) cold items.".format(
                self.RECOMMENDER_NAME, cold_items_mask.sum(), cold_items_mask.sum() / len(cold_items_mask)*100))

        similarity = Compute_Similarity(train, 
                                        shrink=self.shrink, 
                                        topK=self.k,
                                        normalize=self.normalize, 
                                        similarity=self.similarity)
        
        w_sparse = similarity.compute_similarity()
        w_sparse = w_sparse.tocsc()

        self.pred_mat = train.dot(w_sparse).tolil()

    def predict(self, u, i):
        if u >= self.user_num or i >= self.item_num:
            raise ValueError('User and/or item is unkown.')

        return self.pred_mat[u, i]

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make ItemKNN available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat


class UserKNNCF(object):
    def __init__(self, user_num, item_num, maxk=40, shrink=100, 
                 similarity='cosine', normalize=True):
        """
        UserKNN recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        maxk : int, the max similar items number
        shrink : float, shrink similarity value
        similarity : str, way to calculate similarity
        normalize : bool, whether calculate similarity with normalized value
        """
        self.user_num = user_num
        self.item_num = item_num

        self.k = maxk
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.RECOMMENDER_NAME = "UserKNNCFRecommender"

        self.pred_mat = None

    def fit(self, train_set):
        train = self._convert_df(self.user_num, self.item_num, train_set)

        cold_user_mask = np.ediff1d(train.tocsc().indptr) == 0
        if cold_user_mask.any():
            print("{}: Detected {} ({:.2f} %) cold users.".format(
                self.RECOMMENDER_NAME, cold_user_mask.sum(), cold_user_mask.sum()/len(cold_user_mask)*100))

        similarity = Compute_Similarity(train.T, 
                                        shrink=self.shrink, 
                                        topK=self.k,
                                        normalize=self.normalize, 
                                        similarity = self.similarity)

        w_sparse = similarity.compute_similarity()
        w_sparse = w_sparse.tocsc()

        self.pred_mat = w_sparse.dot(train).tolil()

    def predict(self, u, i):
        if u >= self.user_num or i >= self.item_num:
            raise ValueError('User and/or item is unkown.')

        return self.pred_mat[u, i]

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make UserKNN available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat
