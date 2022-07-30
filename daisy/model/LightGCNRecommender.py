'''
@inproceedings{he2020lightgcn,
  title={Lightgcn: Simplifying and powering graph convolution network for recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval},
  pages={639--648},
  year={2020}
}
'''
import torch
import torch.nn as nn

import numpy as np
import scipy.sparse as sp

from daisy.model.AbstractRecommender import GeneralRecommender

class LightGCN(GeneralRecommender):
    def __init__(self, config):
        '''
        LightGCN Recommender Class

        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, embedding dimension
        num_layers : int, number of ego layers
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        '''  
        super(LightGCN, self).__init__(config)

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.topk = config['topk']
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        
        # get this matrix from utils.get_inter_matrix and add it in config
        self.interaction_matrix = config['inter_matrix']

        self.factors = config['factors']
        self.num_layers = config['num_layers']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        
        self.embed_user = nn.Embedding(self.user_num, self.factors)
        self.embed_item = nn.Embedding(self.item_num, self.factors)

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'xavier_uniform'
        self.early_stop = config['early_stop']

        # storage variables for rank evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(self._init_weight)

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

    def get_norm_adj_mat(self):
        '''
        Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        '''
        # build adj matrix
        A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # norm adj matrix
        sum_arr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sum_arr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        ''' Get the embedding of users and items and combine to an new embedding matrix '''
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_embedding, item_embedding = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_embedding, item_embedding

    def calc_loss(self, batch):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)

        embed_user, embed_item = self.forward()

        u_embeddings = embed_user[user]
        pos_embeddings = embed_item[pos_item]
        pos_pred = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)

        u_ego_embeddings = self.embed_user(user)
        pos_ego_embeddings = self.embed_item(pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device)
            loss = self.criterion(pos_pred, label)
            # add regularization term
            loss += self.reg_1 * (u_ego_embeddings.weight.norm(p=1) + pos_ego_embeddings.weight.norm(p=1))
            loss += self.reg_2 * (u_ego_embeddings.weight.norm() + pos_ego_embeddings.weight.norm())

        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_embeddings = embed_item[neg_item]
            neg_pred = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            neg_ego_embeddings = self.item_embedding(neg_item)

            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (u_ego_embeddings.weight.norm(p=1) + pos_ego_embeddings.weight.norm(p=1) + neg_ego_embeddings.weight.norm(p=1))
            loss += self.reg_2 * (u_ego_embeddings.weight.norm() + pos_ego_embeddings.weight.norm() + neg_ego_embeddings.weight.norm())

        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        return loss

    def predict(self, u, i):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embedding = self.restore_user_e[u]
        i_embedding = self.restore_item_e[i]
        pred = torch.matmul(u_embedding, i_embedding.t())

        return pred.cpu()

    def rank(self, test_loader):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            user_emb = self.restore_user_e[us].unsqueeze(dim=1) # batch * 1 * factor
            item_emb = self.restore_item_e[cands_ids].transpose(0, 2, 1) # batch * factor * cand_num
            scores = torch.bmm(user_emb, item_emb).squeeze() # batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user_emb = self.restore_user_e[u] # factor
        items_emb = self.restore_item_e.data # item * factor
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0))

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
