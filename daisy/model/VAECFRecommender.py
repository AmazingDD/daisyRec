'''
@inproceedings{liang2018variational,
  title={Variational autoencoders for collaborative filtering},
  author={Liang, Dawen and Krishnan, Rahul G and Hoffman, Matthew D and Jebara, Tony},
  booktitle={Proceedings of the 2018 world wide web conference},
  pages={689--698},
  year={2018}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from daisy.model.AbstractRecommender import AERecommender


class VAECF(AERecommender):
    def __init__(self, config):
        """
        VAE Recommender Class
        Parameters
        ----------
        mlp_hidden_size : List, Q-net dimension list
        dropout : float, drop out rate
        epochs : int, number of training epochs
        lr : float, learning rate
        latent_dim: size of bottleneck layer
        anneal_cap : float, regularization for KLD
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        early_stop : bool, whether to activate early stop mechanism
        """
        super(VAECF, self).__init__(config)
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.dropout = config['dropout']

        self.layers = config["mlp_hidden_size"] if config['mlp_hidden_size'] is not None else [600]
        self.lat_dim = config['latent_dim']
        self.anneal_cap = config['anneal_cap']
        self.total_anneal_steps = config["total_anneal_steps"]

        self.user_num = config['user_num']
        self.item_num = config['item_num']

        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)
        self.update = 0

        self.encode_layer_dims = [self.item_num] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)
        
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'xavier_normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)
        self.topk = config['topk']

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # core calculation for predicting the real distribution
        else:
            return mu

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)
        h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)

        mu = h[:, :int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2):]

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)

        return z, mu, logvar

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        rating_matrix = self.get_user_rating_matrix(user)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = self.forward(rating_matrix)
        # KL loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) * anneal
        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        loss = ce_loss + kl_loss
        
        return loss

    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)

        rating_matrix = self.get_user_rating_matrix(u)
        scores, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(i)).to(self.device), i]].cpu()

    def rank(self, test_loader):
        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            rating_matrix = self.get_user_rating_matrix(us)
            scores, _, _ = self.forward(rating_matrix) # dimension of scores: batch * item_num
            scores = scores[torch.arange(cands_ids.shape[0]).to(self.device).reshape(-1, 1).expand_as(cands_ids), cands_ids] # batch * item_num -> batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        u = torch.tensor(u, device=self.device)
        rating_matrix = self.get_user_rating_matrix(u)
        scores, _, _ = self.forward(rating_matrix)

        return torch.argsort(scores.view(-1), descending=True)[:self.topk].cpu().numpy()
