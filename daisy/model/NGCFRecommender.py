'''
@inproceedings{wang2019neural,
  title={Neural graph collaborative filtering},
  author={Wang, Xiang and He, Xiangnan and Wang, Meng and Feng, Fuli and Chua, Tat-Seng},
  booktitle={Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval},
  pages={165--174},
  year={2019}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp

from daisy.model.AbstractRecommender import GeneralRecommender

class SparseDropout(nn.Module):
    '''
    This is a Module that execute Dropout on Pytorch sparse tensor.
    '''
    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)

class BiGNN(nn.Module):
    '''
    Propagate a layer of Bi-interaction GNN
        L = D^-1(A)D^-1 is laplace matrix, I is identity matrix
        output = (L+I)EW_1 + LE \otimes EW_2 = (LE + E)W_1 + LE \otimes EW_2
        so I can be never used.
    '''

    def __init__(self, in_dim, out_dim):
        super(BiGNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interact_transform = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, lap_matrix, features):
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interact_transform(inter_feature)

        return inter_part1 + inter_part2

class NGCF(GeneralRecommender):
    def __init__(self, config):
        """
        NGCF Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        node_dropout: float, node dropout ratio
        mess_dropout : float, messsage dropout rate
        hidden_size_list : list, dimension structure of hidden layers, optional.
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        epochs : int, number of training epochs
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(NGCF, self).__init__(config)

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.topk = config['topk']
        self.user_num = config['user_num']
        self.item_num = config['item_num']

        # get this matrix from utils.get_inter_matrix and add it in config
        self.interaction_matrix = config['inter_matrix']

        self.embedding_size = config['factors']
        self.hidden_size_list = config["hidden_size_list"] if config['hidden_size_list'] is not None else [64, 64, 64]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list

        self.node_dropout = config['node_dropout']
        self.message_dropout = config['mess_dropout']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']

        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.embed_user = nn.Embedding(self.user_num, self.embedding_size)
        self.embed_item = nn.Embedding(self.item_num, self.embedding_size)
        self.gnn_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.gnn_layers.append(BiGNN(in_size, out_size))
        
        # storage variables for evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['init_method'] if config['init_method'] != 'default' else 'xavier_normal'
        self.early_stop = config['early_stop']

        # parameters initialization
        self.apply(self._init_weight)

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        # self.eye_matrix = self.get_eye_mat().to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sum_arr = (A > 0).sum(axis=1)
        diag = np.array(sum_arr.flatten())[0] + 1e-7  # add epsilon to avoid divide by zero Warning
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

    def get_eye_mat(self):
        num = self.user_num + self.item_num  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        user_embeddings = self.embed_user.weight
        item_embeddings = self.embed_item.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        A_hat = self.sparse_dropout(self.norm_adj_matrix) if self.node_dropout != 0 else self.norm_adj_matrix
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.gnn_layers:
            all_embeddings = gnn(A_hat, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.user_num, self.item_num])

        return user_all_embeddings, item_all_embeddings

    def calc_loss(self, batch):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = batch[0].to(self.device).long()
        pos_item = batch[1].to(self.device).long()

        embed_user, embed_item = self.forward()

        u_embeddings = embed_user[user]
        pos_embeddings = embed_item[pos_item]
        pos_pred = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)

        u_ego_embeddings = self.embed_user(user)
        pos_ego_embeddings = self.embed_item(pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)
            # add regularization term
            loss += self.reg_1 * (u_ego_embeddings.norm(p=1) + pos_ego_embeddings.norm(p=1))
            loss += self.reg_2 * (u_ego_embeddings.norm() + pos_ego_embeddings.norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device).long()
            neg_embeddings = embed_item[neg_item]
            neg_pred = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            neg_ego_embeddings = self.embed_item(neg_item)

            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (u_ego_embeddings.norm(p=1) + pos_ego_embeddings.norm(p=1) + neg_ego_embeddings.norm(p=1))
            loss += self.reg_2 * (u_ego_embeddings.norm() + pos_ego_embeddings.norm() + neg_ego_embeddings.norm())

        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        return loss

    def predict(self, u, i):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embedding = self.restore_user_e[u]
        i_embedding = self.restore_item_e[i]
        pred = torch.matmul(u_embedding, i_embedding.t())

        return pred.cpu().item()

    def rank(self, test_loader):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            user_emb = self.restore_user_e[us].unsqueeze(dim=1) # batch * 1 * factor
            item_emb = self.restore_item_e[cands_ids].transpose(1, 2) # batch * factor * cand_num
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
