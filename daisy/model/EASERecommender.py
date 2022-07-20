'''
@inproceedings{steck2019embarrassingly,
  title={Embarrassingly shallow autoencoders for sparse data},
  author={Steck, Harald},
  booktitle={The World Wide Web Conference},
  pages={3251--3257},
  year={2019}
}
'''
import numpy as np
import scipy.sparse as sp

from daisy.model.AbstractRecommender import GeneralRecommender


class EASE(GeneralRecommender):
    def __init__(self, config):
        super(EASE, self).__init__(config)
        self.inter_name = config['INTER_NAME']
        self.iid_name = config['IID_NAME']
        self.uid_name = config['UID_NAME']

        self.user_num = config['user_num']
        self.item_num = config['item_num']

        self.reg_weight = config['reg']

        self.topk = config['topk']

    def fit(self, train_set):
        row_ids = train_set[self.uid_name].values
        col_ids = train_set[self.iid_name].values
        values = train_set[self.inter_name].values

        X = sp.csr_matrix((values, (row_ids, col_ids)), shape=(self.user_num, self.item_num)).astype(np.float32)

        G = X.T @ X # item_num * item_num
        G += self.reg_weight * sp.identity(G.shape[0])
        G = G.todense() # why not just use scipy?

        P = np.linalg.inv(G)
        B = -P / np.diag(P) # equation 8 in paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
        np.fill_diagonal(B, 0.)

        self.item_similarity = B # item_num * item_num
        self.interaction_matrix = X # user_num * item_num

    def predict(self, u, i):
        self.interaction_matrix[u, :].multiply(self.item_similarity[:, i].T).sum(axis=1).getA1()

    def rank(self, test_loader):
        rec_ids = np.array([])

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy()

            slims = np.expand_dims(self.interaction_matrix[us, :].todense(), axis=1) # batch * item_num -> batch * 1* item_num
            sims = self.item_similarity[cands_ids, :].transpose(0, 2, 1) # batch * cand_num * item_num -> batch * item_num * cand_num
            scores = np.einsum('BNi,BiM -> BNM', slims, sims).squeeze() # batch * 1 * cand_num -> batch * cand_num
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[:, rank_ids]

            rec_ids = np.vstack([rec_ids, rank_list])

        return rec_ids

    def full_rank(self, u):
        r = self.interaction_matrix[u, :] @ self.item_similarity
        scores = np.array(r).flatten()

        return np.argsort(-scores)[:self.topk]
