'''
@inproceedings{kang2016top,
  title={Top-n recommender system via matrix completion},
  author={Kang, Zhao and Peng, Chong and Cheng, Qiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={30},
  number={1},
  year={2016}
}
'''
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

from daisy.model.AbstractRecommender import GeneralRecommender


class PureSVD(GeneralRecommender):
    def __init__(self, config):
        """
        PureSVD Recommender
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, latent factor number
        """
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.factors = config['factors']

        self.user_vec = None
        self.item_vec = None

        self.topk = config['topk']

    def fit(self, train_set):
        self.logger.info(" Computing SVD decomposition...")
        train_set = self._convert_df(self.user_num, self.item_num, train_set)
        self.logger.info('Finish build train matrix for decomposition')
        U, sigma, Vt = randomized_svd(train_set,
                                      n_components=self.factors,
                                      random_state=2019)
        s_Vt = sp.diags(sigma) * Vt

        self.user_vec = U
        self.item_vec = s_Vt.T
        self.logger.info('Done!')

    def _convert_df(self, user_num, item_num, df):
        """Process Data to make WRMF available"""
        ratings = list(df['rating'])
        rows = list(df['user'])
        cols = list(df['item'])
        mat = sp.csr_matrix((ratings, (rows, cols)), shape=(user_num, item_num))

        return mat

    def predict(self, u, i):
        return self.user_vec[u, :].dot(self.item_vec[i, :])

    def rank(self, test_loader):
        rec_ids = np.array([])

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy() # batch * cand_num

            user_emb = np.expand_dims(self.user_vec[us, :], axis=1) # batch * factor -> batch * 1 * factor
            items_emb = self.item_vec[cands_ids, :].transpose(0, 2, 1)  # batch * cand_num * factor -> batch * factor * cand_num
            scores = np.einsum('BNi,BiM -> BNM', user_emb, items_emb).squeeze() # batch * 1 * cand_num -> batch * cand_num
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[:, rank_ids]
            
            rec_ids = np.vstack([rec_ids, rank_list])

        return rec_ids

    def full_rank(self, u):
        scores = self.user_vec[u, :].dot(self.item_vec.T) #  (item_num,)

        return np.argsort(-scores)[:self.topk]

