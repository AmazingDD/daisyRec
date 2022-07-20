'''
@inproceedings{ji2020re,
  title={A re-visit of the popularity baseline in recommender systems},
  author={Ji, Yitong and Sun, Aixin and Zhang, Jie and Li, Chenliang},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1749--1752},
  year={2020}
}
'''
import torch
import numpy as np

from daisy.model.AbstractRecommender import GeneralRecommender


class MostPop(GeneralRecommender):
    def __init__(self, config):
        """
        Most Popular Recommender
        Parameters
        ----------
        """
        self.item_num = config['item_num']
        self.item_cnt_ref = np.zeros(self.item_num)
        self.topk = config['topk']

    def fit(self, train_set):
        item_cnt = train_set['item'].size()
        idx, cnt = item_cnt.index, item_cnt.values
        self.item_cnt_ref[idx] = cnt
        self.item_score =  self.item_cnt_ref / (1 + self.item_cnt_ref)

    def predict(self, u, i):
        return self.item_score[i]

    def rank(self, test_loader):
        item_score = torch.tensor(self.item_score, device=self.device)

        rec_ids = torch.tensor([], device=self.device)
        for _, cands_ids in test_loader:
            cands_ids = cands_ids.to(self.device)
            scores = item_score[cands_ids] # batch_size * cand_num
            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        return np.argsort(-self.item_score)[:self.topk]
