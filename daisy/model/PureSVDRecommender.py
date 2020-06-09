import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


class PureSVD(object):
    def __init__(self, user_num, item_num, factors=150):
        """
        PureSVD Recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, latent factor number
        """
        self.user_num = user_num
        self.item_num = item_num
        self.factors = factors

        self.user_vec = None
        self.item_vec = None

    def fit(self, df):
        print(" Computing SVD decomposition...")
        train_set = self._convert_df(self.user_num, self.item_num, df)
        print('Finish build train matrix for decomposition')
        U, sigma, Vt = randomized_svd(train_set,
                                      n_components=self.factors,
                                      random_state=2019)
        s_Vt = sp.diags(sigma) * Vt

        self.user_vec = U
        self.item_vec = s_Vt.T
        print('Done!')

    def predict(self, u, i):
        return self.user_vec[u, :].dot(self.item_vec[i, :])

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make WRMF available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csr_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat
