'''
@inproceedings{steck2019embarrassingly,
  title={Embarrassingly shallow autoencoders for sparse data},
  author={Steck, Harald},
  booktitle={The World Wide Web Conference},
  pages={3251--3257},
  year={2019}
}
'''
from typing import TypedDict, Iterable

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


class EaseConfig(TypedDict):
    user_num: int
    item_num: int
    reg: float

class EASE:
    def __init__(self, config: EaseConfig) -> None:
        super(EASE, self).__init__(config)
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.reg_weight = config['reg']

    def fit(self, users: npt.NDArray[np.int64], items: npt.NDArray[np.int64], values: npt.NDArray[np.float64]) -> None:
        """
        if train_set is pd.DataFrame
        users = train_set[self.uid_name].values
        items = train_set[self.iid_name].values
        values = train_set[self.inter_name].values
        
        Args:
            users (npt.NDArray[np.int64]): user ids
            items (npt.NDArray[np.int64]): item ids
            values (npt.NDArray[Union[np.float64]]): interaction value (For example: movie rating) 
        """

        
        row_ids = users
        col_ids = items
        
        X = sp.csr_matrix((values, (row_ids, col_ids)), shape=(self.user_num, self.item_num)).astype(np.float32)

        G = X.T @ X # item_num * item_num
        G += self.reg_weight * sp.identity(G.shape[0])
        G = G.todense() # why not just use scipy?

        P = np.linalg.inv(G)
        B = -P / np.diag(P) # equation 8 in paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
        np.fill_diagonal(B, 0.)

        self.item_similarity = B # item_num * item_num
        self.item_similarity = np.array(self.item_similarity)
        self.interaction_matrix = X # user_num * item_num

    def predict(self, u: Iterable[int], i: Iterable[int]) -> float:
        return self.interaction_matrix[u, :].multiply(self.item_similarity[:, i].T).sum(axis=1).getA1()[0]

    def rank(self, us: npt.NDArray[np.int64], cands_ids: npt.NDArray[np.int64], topk: int) -> npt.NDArray[np.int64]:
        slims = np.expand_dims(self.interaction_matrix[us, :].todense(), axis=1) # batch * item_num -> batch * 1* item_num
        sims = self.item_similarity[cands_ids, :].transpose(0, 2, 1) # batch * cand_num * item_num -> batch * item_num * cand_num
        scores = np.einsum('BNi,BiM -> BNM', slims, sims).squeeze(axis=1) # batch * 1 * cand_num -> batch * cand_num
        rank_ids = np.argsort(-scores)[:, :topk]
        rec_ids = cands_ids[np.repeat(np.arange(len(rank_ids)).reshape(-1, 1), rank_ids.shape[1], axis=1), rank_ids]
        
        return rec_ids

    def full_rank(self, u: Iterable[int], topk: int) -> npt.NDArray[np.int64]:
        scores = self.interaction_matrix[u, :] @ self.item_similarity
        return np.argsort(-scores)[:, :topk]
