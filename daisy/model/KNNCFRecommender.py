'''
@inproceedings{sarwar2001item,
  title={Item-based collaborative filtering recommendation algorithms},
  author={Sarwar, Badrul and Karypis, George and Konstan, Joseph and Riedl, John},
  booktitle={Proceedings of the 10th international conference on World Wide Web},
  pages={285--295},
  year={2001}
}
@inproceedings{aiolli2013efficient,
  title={Efficient top-n recommendation for very large scale binary rated datasets},
  author={Aiolli, Fabio},
  booktitle={Proceedings of the 7th ACM conference on Recommender systems},
  pages={273--280},
  year={2013}
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

from daisy.model.AbstractRecommender import GeneralRecommender


def convert_df(user_num, item_num, df):
    """ 
    Convert DataFrame to make matrix to make similarity calculation available
    """
    ratings = list(df['rating'])
    rows = list(df['user'])
    cols = list(df['item'])

    mat = sp.csc_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

    return mat

def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    """
    if format == 'csc' and not isinstance(X, sp.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sp.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sp.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sp.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sp.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sp.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sp.lil_matrix):
        return X.tolil().astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sp.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)

class Similarity:
    def __init__(self, data_matrix, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine", row_weights=None):  
        '''
        Computes the cosine similarity on the columns of data_matrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.

        Parameters
        ----------
        data_matrix : _type_
            _description_
        topK : int, optional
            the K-nearest number, by default 100
        shrink : int, optional
            smooth factor for denomitor when computing, by default 0
        normalize : bool, optional
            If True divide the dot product by the product of the norms, by default True
        asymmetric_alpha : float, optional
            Coefficient alpha for the asymmetric cosine, by default 0.5
        similarity : str, optional
            "cosine"        computes Cosine similarity
            "adjusted"      computes Adjusted Cosine, removing the average of the users
            "asymmetric"    computes Asymmetric Cosine
            "pearson"       computes Pearson Correlation, removing the average of the items
            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
            "dice"          computes Dice similarity for binary interactions
            "tversky"       computes Tversky similarity for binary interactions
            "tanimoto"      computes Tanimoto coefficient for binary interactions, 
            by default "cosine"
        row_weights : array, optional
            Multiply the values in each row by a specified value. Array, by default None
        '''
        super(Similarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = data_matrix.shape
        self.topk = min(topK, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.data_matrix = data_matrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError(f"value for parameter 'similarity' not recognized. Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto', 'dice', 'tversky'. Passed value was '{similarity}'")

        self.use_row_weights = False

        if row_weights is not None:
            if data_matrix.shape[0] != len(row_weights):
                raise ValueError(f"provided row_weights and data_matrix have different number of rows. Col_weights has {len(row_weights)} columns, data_matrix has {data_matrix.shape[0]}.")

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sp.diags(self.row_weights)

            self.data_matrix_weighted = self.data_matrix.T.dot(self.row_weights_diag).T

    def apply_adjusted_cosine(self):
        """
        Remove from every data point the average for the corresponding row
        """

        self.data_matrix = check_matrix(self.data_matrix, 'csr')

        interactions_per_row = np.diff(self.data_matrix.indptr)

        nonzero_rows = interactions_per_row > 0
        sum_per_row = np.asarray(self.data_matrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sum_per_row)
        rowAverage[nonzero_rows] = sum_per_row[nonzero_rows] / interactions_per_row[nonzero_rows]

        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        block_size = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + block_size)

            self.data_matrix.data[self.data_matrix.indptr[start_row]:self.data_matrix.indptr[end_row]] -= np.repeat(rowAverage[start_row:end_row], interactions_per_row[start_row:end_row])

            start_row += block_size

    def apply_pearson_correlation(self):
        """
        Remove from every data point the average for the corresponding column
        """

        self.data_matrix = check_matrix(self.data_matrix, 'csc')

        interactions_per_col = np.diff(self.data_matrix.indptr)

        nonzero_cols = interactions_per_col > 0
        sum_per_col = np.asarray(self.data_matrix.sum(axis=0)).ravel()

        col_average = np.zeros_like(sum_per_col)
        col_average[nonzero_cols] = sum_per_col[nonzero_cols] / interactions_per_col[nonzero_cols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        block_size = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + block_size)

            self.data_matrix.data[self.data_matrix.indptr[start_col]:self.data_matrix.indptr[end_col]] -= np.repeat(col_average[start_col:end_col], interactions_per_col[start_col:end_col])

            start_col += block_size

    def use_boolean_interactions(self):
        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        block_size = 1000

        while end_pos < len(self.data_matrix.data):
            end_pos = min(len(self.data_matrix.data), end_pos + block_size)

            self.data_matrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += block_size

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processed_items = 0

        if self.adjusted_cosine:
            self.apply_adjusted_cosine()

        elif self.pearson_correlation:
            self.apply_pearson_correlation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.use_boolean_interactions()

        # We explore the matrix column-wise
        self.data_matrix = check_matrix(self.data_matrix, 'csc')

        # Compute sum of squared values to be used in normalization
        sum_of_squared = np.array(self.data_matrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sum_of_squared = np.sqrt(sum_of_squared)

        if self.asymmetric_cosine:
            sum_of_squared_to_1_minus_alpha = np.power(sum_of_squared, 2 * (1 - self.asymmetric_alpha))
            sum_of_squared_to_alpha = np.power(sum_of_squared, 2 * self.asymmetric_alpha)

        self.data_matrix = check_matrix(self.data_matrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:
            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.data_matrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.data_matrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.data_matrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                column_index = col_index_in_block + start_col_block
                this_column_weights[column_index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    if self.asymmetric_cosine:
                        denominator = sum_of_squared_to_alpha[column_index] * sum_of_squared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sum_of_squared[column_index] * sum_of_squared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sum_of_squared[column_index] + sum_of_squared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sum_of_squared[column_index] + sum_of_squared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + (sum_of_squared[column_index] - this_column_weights) * self.tversky_alpha + (sum_of_squared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                # Sort indices and select topk
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.topk - 1)[0:self.topk]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                not_zeros_mask = this_column_weights[top_k_idx] != 0.0
                num_not_zeros = np.sum(not_zeros_mask)

                values.extend(this_column_weights[top_k_idx][not_zeros_mask])
                rows.extend(top_k_idx[not_zeros_mask])
                cols.extend(np.ones(num_not_zeros) * column_index)

            # Add previous block size
            processed_items += this_block_size

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                column_per_sec = processed_items / (time.time() - start_time + 1e-9)

                self.logger.info(f"Similarity column {processed_items} ( {processed_items / (end_col_local - start_col_local) * 100:2.0f} % ), {column_per_sec:.2f} column/sec, elapsed time {(time.time() - start_time) / 60:.2f} min")

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            start_col_block += block_size

        w_sparse = sp.csr_matrix((values, (rows, cols)),
                                 shape=(self.n_columns, self.n_columns),
                                 dtype=np.float32)

        return w_sparse


class ItemKNNCF(GeneralRecommender):
    def __init__(self, config):
        """
        ItemKNN recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        maxk : int, the max nearest similar items number
        shrink : float, shrink similarity value
        similarity : str, way to calculate similarity
                    "cosine"        computes Cosine similarity
                    "adjusted"      computes Adjusted Cosine, removing the average of the users
                    "asymmetric"    computes Asymmetric Cosine
                    "pearson"       computes Pearson Correlation, removing the average of the items
                    "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                    "dice"          computes Dice similarity for binary interactions
                    "tversky"       computes Tversky similarity for binary interactions
                    "tanimoto"      computes Tanimoto coefficient for binary interactions, 
                    by default "cosine"
        normalize : bool, whether calculate similarity with normalized value
        """
        self.user_num = config['user_num']
        self.item_num = config['item_num']

        self.k = config['maxk']
        self.shrink = config['shrink']
        self.normalize = config['normalize']
        self.similarity = config['similarity']

        self.topk = config['topk']

        self.pred_mat = None

    def fit(self, train_set):
        train = convert_df(self.user_num, self.item_num, train_set)

        cold_items_mask = np.ediff1d(train.tocsc().indptr) == 0
        if cold_items_mask.any():
            self.logger.info(f"ItemKNNCFRecommender: Detected {cold_items_mask.sum()} ({cold_items_mask.sum() / len(cold_items_mask) * 100:.2f} %) cold items.")

        similarity = Similarity(train, 
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

    def rank(self, test_loader):
        rec_ids = np.array([])

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy()
            scores = self.pred_mat[us, cands_ids].A
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[:, rank_ids]

            rec_ids = np.vstack([rec_ids, rank_list])

        return rec_ids

    def full_rank(self, u):
        scores = self.pred_mat[u, :].A.squeeze()

        return np.argsort(-scores)[:self.topk]

class UserKNNCF(GeneralRecommender):
    def __init__(self, config):
        """
        UserKNN recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        maxk : int, the max similar items number
        shrink : float, shrink similarity value
        similarity : str, way to calculate similarity
                    "cosine"        computes Cosine similarity
                    "adjusted"      computes Adjusted Cosine, removing the average of the users
                    "asymmetric"    computes Asymmetric Cosine
                    "pearson"       computes Pearson Correlation, removing the average of the items
                    "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                    "dice"          computes Dice similarity for binary interactions
                    "tversky"       computes Tversky similarity for binary interactions
                    "tanimoto"      computes Tanimoto coefficient for binary interactions, 
                    by default "cosine"
        normalize : bool, whether calculate similarity with normalized value
        """
        self.user_num = config['user_num']
        self.item_num = config['item_num']

        self.k = config['maxk']
        self.shrink = config['shrink']
        self.normalize = config['normalize']
        self.similarity = config['similarity']

        self.pred_mat = None

    def fit(self, train_set):
        train = convert_df(self.user_num, self.item_num, train_set)

        cold_user_mask = np.ediff1d(train.tocsc().indptr) == 0
        if cold_user_mask.any():
            self.logger.info(f"UserKNNCFRecommender: Detected {cold_user_mask.sum()} ({cold_user_mask.sum()/len(cold_user_mask) * 100:.2f} %) cold users.")

        similarity = Similarity(train.T, 
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

    def rank(self, test_loader):
        rec_ids = np.array([])

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy()
            scores = self.pred_mat[us, cands_ids].A
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[:, rank_ids]

            rec_ids = np.vstack([rec_ids, rank_list])

        return rec_ids

    def full_rank(self, u):
        scores = self.pred_mat[u, :].A.squeeze()

        return np.argsort(-scores)[:self.topk]
