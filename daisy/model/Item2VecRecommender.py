'''
@inproceedings{barkan2016item2vec,
  title={Item2vec: neural item embedding for collaborative filtering},
  author={Barkan, Oren and Koenigstein, Noam},
  booktitle={2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
'''
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender

class Item2Vec(GeneralRecommender):
    def __init__(self, config):
        '''
        Item2Vec Recommender class

        Parameters
        ----------
        factors : int
            item embedding dimension
        lr : float
            learning rate
        epochs : int
            No. of training iterations
        '''
        super(Item2Vec, self).__init__(config)

        self.user_embedding = nn.Embedding(config['user_num'], config['factors'])
        self.ur = config['train_ur']

        self.shared_embedding = nn.Embedding(config['item_num'], config['factors'])
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.out_act = nn.Sigmoid()

        # default loss function for item2vec is cross-entropy
        self.loss_type = 'CL'
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)

    def forward(self, target_i, context_j):
        target_emb = self.shared_embedding(target_i) # batch_size * embedding_size
        context_emb = self.shared_embedding(context_j) # batch_size * embedding_size
        output = torch.sum(target_emb * context_emb, dim=1)
        output = self.out_act(output)

        return output.view(-1)

    def fit(self, train_loader):
        super().fit(train_loader)

        self.logger.info('Start building user embedding...')
        for u in self.ur.keys():
            uis = torch.tensor(list(self.ur[u]), device=self.device)
            self.user_embedding.weight.data[u] = self.shared_embedding(uis).sum(dim=0)

    def calc_loss(self, batch):
        target_i = batch[0].to(self.device)
        context_j = batch[1].to(self.device)
        label = batch[2].to(self.device)
        prediction = self.forward(target_i, context_j)
        loss = self.criterion(prediction, label)
        
        return loss
    
    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        
        user_emb = self.user_embedding(u)
        item_emb = self.shared_embedding(i)
        pred = (user_emb * item_emb).sum(dim=-1)
        
        return pred.cpu()

    def rank(self, test_loader):
        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            user_emb = self.user_embedding(us).unsqueeze(dim=1) # batch * factor -> batch * 1 * factor
            item_emb = self.shared_embedding(cands_ids).transpose(0, 2, 1) # batch * cand_num * factor -> batch * factor * cand_num 
            scores = torch.bmm(user_emb, item_emb).squeeze() # batch * 1 * cand_num -> batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        u = torch.tensor(u, self.device)

        user_emb = self.user_embedding(u)
        items_emb = self.shared_embedding.weight 
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0)) #  (item_num,)

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
