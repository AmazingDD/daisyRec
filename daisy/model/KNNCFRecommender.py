import os
import heapq
import numpy as np
import scipy.sparse as sp
from enum import Enum
from collections import defaultdict

from daisy.model.simlib_python import Compute_Similarity_Python

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
                from daisy.model.simlib_cython import Compute_Similarity_Cython
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
    """ ItemKNN recommender"""
    def __init__(self, user_num, item_num, maxk=40, shrink=100, 
                 similarity='cosine', min_k=1, normalize=True,
                 tune_or_not=False, serial='ml-100k-origin-loo-0-cosine'):
        self.user_num = user_num
        self.item_num = item_num

        self.k = maxk
        self.min_k = min_k
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity

        self.RECOMMENDER_NAME = "ItemKNNCFRecommender"

        self.tune_or_not = tune_or_not
        self.serial = serial

        if not os.path.exists('./tmp/itemknncf/sim_matrix/'):
            os.makedirs('./tmp/itemknncf/sim_matrix/')

    def fit(self, train_set):
        self.yr = defaultdict(list)
        for _, row in train_set.iterrows():
            self.yr[int(row['user'])].append((int(row['item']), row['rating']))

        train = self._convert_df(self.user_num, self.item_num, train_set)
        
        cold_items_mask = np.ediff1d(train.tocsc().indptr) == 0

        if cold_items_mask.any():
            print("{}: Detected {} ({:.2f} %) cold items.".format(
                self.RECOMMENDER_NAME, cold_items_mask.sum(), cold_items_mask.sum()/len(cold_items_mask)*100))

        similarity = Compute_Similarity(train, 
                                        shrink=self.shrink, 
                                        topK=self.k,
                                        normalize=self.normalize, 
                                        similarity = self.similarity)
        
        W_sparse = similarity.compute_similarity()
        W_sparse = W_sparse.tocsc()

        self.pred_mat = train.dot(W_sparse).tolil()

    def predict(self, u, i):
        if u >= self.user_num or i >= self.item_num:
            raise ValueError('User and/or item is unkown.')

        return self.pred_mat[u, i]

    def _convert_df(self, user_num, item_num, df):
        '''Process Data to make ItemKNN available'''
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat

class UserKNNCF(object):
    """ UserKNN recommender"""
    def __init__(self, user_num, item_num, maxk=40, shrink=100, 
                 similarity='cosine', min_k=1, normalize=True, 
                 tune_or_not=False, serial='ml-100k-origin-loo-0-cosine'):
        self.user_num = user_num
        self.item_num = item_num

        self.k = maxk
        self.min_k = min_k
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity

        self.RECOMMENDER_NAME = "UserKNNCFRecommender"

        self.tune_or_not = tune_or_not
        self.serial = serial

        if not os.path.exists('./tmp/userknncf/sim_matrix/'):
            os.makedirs('./tmp/userknncf/sim_matrix/')

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

        W_sparse = similarity.compute_similarity()
        W_sparse = W_sparse.tocsc()

        self.pred_mat = W_sparse.dot(train).tolil()

    def predict(self, u, i):
        if u >= self.user_num or i >= self.item_num:
            raise ValueError('User and/or item is unkown.')

        return self.pred_mat[u, i]

    def _convert_df(self, user_num, item_num, df):
        '''Process Data to make UserKNN available'''
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat