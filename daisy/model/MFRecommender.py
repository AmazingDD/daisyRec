'''
@article{koren2009matrix,
  title={Matrix factorization techniques for recommender systems},
  author={Koren, Yehuda and Bell, Robert and Volinsky, Chris},
  journal={Computer},
  volume={42},
  number={8},
  pages={30--37},
  year={2009},
  publisher={IEEE}
}
@article{rendle2012bpr,
  title={BPR: Bayesian personalized ranking from implicit feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:1205.2618},
  year={2012}
}
'''

import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender

class MF(GeneralRecommender):
    def __init__(self, config):
        """
        Matrix Factorization Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(MF, self).__init__(config)
        
        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.epochs = config['epochs']

        self.topk = config['topk']

        self.embed_user = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item = nn.Embedding(config['item_num'], config['factors'])

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'sgd'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        pred = (embed_user * embed_item).sum(dim=-1)

        return pred

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device)
            loss = self.criterion(pos_pred, label)

            # add regularization term
            loss += self.reg_1 * (self.embed_item(pos_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).weight.norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            # add regularization term
            loss += self.reg_1 * (self.embed_item(pos_item).weight.norm(p=1) + self.embed_item(neg_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).weight.norm() + self.embed_item(neg_item).weight.norm())
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        # add regularization term
        loss += self.reg_1 * (self.embed_user(user).weight.norm(p=1))
        loss += self.reg_2 * (self.embed_user(user).weight.norm())

        return loss

    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        pred = self.forward(u, i).cpu()
        
        return pred

    def rank(self, test_loader):
        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            user_emb = self.embed_user(us).unsqueeze(dim=1) # batch * factor -> batch * 1 * factor
            item_emb = self.embed_item(cands_ids).transpose(0, 2, 1) # batch * cand_num * factor -> batch * factor * cand_num 
            scores = torch.bmm(user_emb, item_emb).squeeze() # batch * 1 * cand_num -> batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()


    def full_rank(self, u):
        u = torch.tensor(u, self.device)

        user_emb = self.embed_user(u)
        items_emb = self.embed_item.weight 
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0)) #  (item_num,)

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()

