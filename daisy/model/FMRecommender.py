'''
@inproceedings{rendle2010factorization,
  title={Factorization machines},
  author={Rendle, Steffen},
  booktitle={2010 IEEE International conference on data mining},
  pages={995--1000},
  year={2010},
  organization={IEEE}
}
'''
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender

class FM(GeneralRecommender):
    def __init__(self, config):
        """
        Factorization Machine Recommender Class
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
        super(FM, self).__init__(config)

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']

        self.embed_user = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item = nn.Embedding(config['item_num'], config['factors'])

        self.u_bias = nn.Embedding(config['user_num'], 1)
        self.i_bias = nn.Embedding(config['item_num'], 1)

        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'sgd'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)
        nn.init.constant_(self.u_bias.weight, 0.0)
        nn.init.constant_(self.i_bias.weight, 0.0)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        pred = (embed_user * embed_item).sum(dim=-1, keepdim=True)
        pred += self.u_bias(user) + self.i_bias(item) + self.bias_

        return pred.view(-1)

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device)
            loss = self.criterion(pos_pred, label)

            loss += self.reg_1 * (self.embed_item(pos_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).weight.norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

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
            scores += self.u_bias(us) + self.i_bias(cands_ids).squeeze() + self.bias_

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
        scores += self.u_bias(u) + self.i_bias.weight + self.bias_

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
