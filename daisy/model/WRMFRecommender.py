import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from scipy.sparse.linalg import spsolve


class WRMF(object):
    def __init__(self, 
                 user_num, 
                 item_num, 
                 df, 
                 factors=20,
                 epochs=10, 
                 alpha=40,
                 reg_2=0.1,  
                 seed=2019):
        """
        WRMF Recommender Class
        Parameters
        ----------
        user_num : int, number of users
        item_num : int, number of items
        df : pd.DataFrame, rating data frame
        factors : latent factor number
        epochs : number of training epochs
        alpha : amplify coefficients
        reg_2 : float, second-order regularization term
        seed : random seed
        """
        train_set = self._convert_df(user_num, item_num, df)

        self.epochs = epochs
        self.rstate = np.random.RandomState(seed)
        self.C = alpha * train_set
        self.user_num, self.item_num = user_num, item_num

        self.X = sp.csr_matrix(self.rstate.normal(scale=0.01, 
                                                  size=(user_num, factors)))
        self.Y = sp.csr_matrix(self.rstate.normal(scale=0.01, 
                                                  size=(item_num, factors)))
        self.X_eye = sp.eye(user_num)
        self.Y_eye = sp.eye(item_num)
        self.lambda_eye = reg_2 * sp.eye(factors)

        self.user_vec, self.item_vec, self.pred_mat = None, None, None

    def fit(self):
        for _ in tqdm(range(self.epochs)):
            yTy = self.Y.T.dot(self.Y)
            xTx = self.X.T.dot(self.X)
            for u in range(self.user_num):
                Cu = self.C[u, :].toarray()
                Pu = Cu.copy()
                Pu[Pu != 0] = 1
                CuI = sp.diags(Cu, [0])
                yTCuIY = self.Y.T.dot(CuI).dot(self.Y)
                yTCuPu = self.Y.T.dot(CuI + self.Y_eye).dot(Pu.T)
                self.X[u] = spsolve(yTy + yTCuIY + self.lambda_eye, yTCuPu)
            for i in range(self.item_num):
                Ci = self.C[:, i].T.toarray()
                Pi = Ci.copy()
                Pi[Pi != 0] = 1
                CiI = sp.diags(Ci, [0])
                xTCiIX = self.X.T.dot(CiI).dot(self.X)
                xTCiPi = self.X.T.dot(CiI + self.X_eye).dot(Pi.T)
                self.Y[i] = spsolve(xTx + xTCiIX + self.lambda_eye, xTCiPi)

        self.user_vec, self.item_vec = self.X, self.Y.T

        # complete prediction matrix in fit process and save time for get rank list
        pred_mat = self.user_vec.dot(self.item_vec)
        self.pred_mat = pred_mat.A

    def predict(self, u, i):
        prediction = self.pred_mat[u, i]
        return prediction

    def _convert_df(self, user_num, item_num, df):
        '''Process Data to make WRMF available'''
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])

        mat = sp.csr_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat
