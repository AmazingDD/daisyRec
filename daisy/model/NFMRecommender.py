'''
@inproceedings{he2017neural,
  title={Neural factorization machines for sparse predictive analytics},
  author={He, Xiangnan and Chua, Tat-Seng},
  booktitle={Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval},
  pages={355--364},
  year={2017}
}
'''
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender

class NFM(GeneralRecommender):
    def __init__(self, config):
        """
        NFM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        act_function : str, activation function for hidden layer
        num_layers : int, number of hidden layers
        batch_norm : bool, whether to normalize a batch of data
        dropout : float, dropout rate
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
        super(NFM, self).__init__(config)
        self.factors = config['factors']
        self.act_function = config['act_function']
        self.num_layers = config['num_layers']
        self.batch_norm = config['batch_norm']
        self.dropout = config['dropout']

        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.epochs = config['epochs']

        self.loss_type = config['loss_type']
        self.initializer = config['init_method'] if config['init_method'] != 'default' else 'xavier_normal'
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'sgd'
        self.early_stop = config['early_stop']
        self.topk = config['topk']

        self.embed_user = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item = nn.Embedding(config['item_num'], config['factors'])

        self.u_bias = nn.Embedding(config['user_num'], 1)
        self.i_bias = nn.Embedding(config['item_num'], 1)

        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(config['factors']))
        FM_modules.append(nn.Dropout(self.dropout))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_modules = []
        in_dim = config['factors']
        for _ in range(self.num_layers):  # dim
            out_dim = in_dim # dim
            MLP_modules.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_modules.append(nn.BatchNorm1d(out_dim))

            if self.act_function == 'relu':
                MLP_modules.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_modules.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_modules.append(nn.Tanh())

            MLP_modules.append(nn.Dropout(self.dropout))
        self.deep_layers = nn.Sequential(*MLP_modules)
        predict_size = config['factors']  # layers[-1] if layers else factors

        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight()

    def _init_weight(self):
        self.initializer_config[self.initializer](self.embed_user.weight, **self.initializer_param_config[self.initializer])
        self.initializer_config[self.initializer](self.embed_item.weight, **self.initializer_param_config[self.initializer])
        nn.init.constant_(self.u_bias.weight, 0.0)
        nn.init.constant_(self.i_bias.weight, 0.0)

        # for deep layers
        if self.num_layers > 0:  # len(self.layers)
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    self.initializer_config[self.initializer](m.weight, **self.initializer_param_config[self.initializer])
            self.initializer_config[self.initializer](self.prediction.weight, **self.initializer_param_config[self.initializer])
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        fm = embed_user * embed_item
        fm = self.FM_layers(fm)

        if self.num_layers:
            fm = self.deep_layers(fm)

        fm += self.u_bias(user) + self.i_bias(item) + self.bias_
        pred = self.prediction(fm)

        return pred.view(-1)

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)

            loss += self.reg_1 * (self.embed_item(pos_item).norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (self.embed_item(pos_item).norm(p=1) + self.embed_item(neg_item).norm(p=1))
            loss += self.reg_2 * (self.embed_item(pos_item).norm() + self.embed_item(neg_item).norm())
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        loss += self.reg_1 * (self.embed_user(user).norm(p=1))
        loss += self.reg_2 * (self.embed_user(user).norm())

        return loss

    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        pred = self.forward(u, i).cpu().item()

        return pred

    def rank(self, test_loader):
        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            user_emb = self.embed_user(us).unsqueeze(dim=1) # batch * factor -> batch * 1 * factor 
            item_emb = self.embed_item(cands_ids) # batch * cand_num * factor

            fm = user_emb * item_emb # batch * cand_num * factor

            batch_dim, cand_dim, factor_dim = fm.size()
            fm = fm.reshape(batch_dim * cand_dim, factor_dim)

            fm = self.FM_layers(fm) # (batch * cand_num) * factor
            if self.num_layers:
                fm = self.deep_layers(fm) # (batch * cand_num) * factor
            
            fm = fm.reshape(batch_dim, cand_dim, factor_dim) # batch * cand_num * factor

            ubs = self.u_bias(us)
            ubs = ubs.repeat_interleave(cand_dim * factor_dim, dim=1).reshape(-1, cand_dim, factor_dim)
            ibs = self.i_bias(cands_ids)

            fm += ubs+ ibs + self.bias_ # batch * cand_num * factor
            scores = self.prediction(fm).squeeze() # batch * cand_num * 1 -> # batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        u = torch.tensor(u, device=self.device)

        user_emb = self.embed_user(u)  # factor
        items_emb = self.embed_item.weight  # item_num * factor

        fm = user_emb * items_emb  # item_num * factor
        fm = self.FM_layers(fm) # item_num * factor
        if self.num_layers:
            fm = self.deep_layers(fm) # item_num * factor
        fm += self.u_bias(u) + self.i_bias.weight + self.bias_
        scores = self.prediction(fm).squeeze() # item_num

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()

        
