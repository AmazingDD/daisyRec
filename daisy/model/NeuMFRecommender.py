'''
@inproceedings{he2017neural,
  title={Neural collaborative filtering},
  author={He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  booktitle={Proceedings of the 26th international conference on world wide web},
  pages={173--182},
  year={2017}
}
'''
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender
from daisy.utils.config import initializer_param_config, initializer_config


class NeuMF(GeneralRecommender):
    def __init__(self, config):
        """
        NeuMF Recommender Class, it can be seperate as: GMF and MLP
        Parameters
        ----------
        user_num : int, number of users;
        item_num : int, number of items;
        factors : int, the number of latent factor
        num_layers : int, number of hidden layers
        dropout : float, dropout rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        model_name : str, model name
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        GMF_model : Object, pre-trained GMF weights;
        MLP_model : Object, pre-trained MLP weights.
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(NeuMF, self).__init__(config)

        self.lr = config['lr']
        self.epochs = config['epochs']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']

        self.dropout = config['dropout']
        self.model = config['model_name']
        self.GMF_model = config['GMF_model']
        self.MLP_model = config['MLP_model']

        self.embed_user_GMF = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item_GMF = nn.Embedding(config['item_num'], config['factors'])

        self.embed_user_MLP = nn.Embedding(config['user_num'], config['factors'] * (2 ** (config['num_layers'] - 1)))
        self.embed_item_MLP = nn.Embedding(config['item_num'], config['factors'] * (2 ** (config['num_layers'] - 1)))

        MLP_modules = []
        for i in range(config['num_layers']):
            input_size = config['factors'] * (2 ** (config['num_layers'] - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = config['factors']
        else:
            predict_size = config['factors'] * 2

        self.predict_layer = nn.Linear(predict_size, 1)

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['initializer'] if config['initializer'] != 'default' else 'xavier_normal'
        self.early_stop = config['early_stop']

        self._init_weight()

    def _init_weight(self):
        if not self.model == 'NeuMF-pre':
            initializer_config[self.initializer](self.embed_user_GMF.weight, **initializer_param_config[self.initializer])
            initializer_config[self.initializer](self.embed_item_GMF.weight, **initializer_param_config[self.initializer])
            initializer_config[self.initializer](self.embed_user_MLP.weight, **initializer_param_config[self.initializer])
            initializer_config[self.initializer](self.embed_item_MLP.weight, **initializer_param_config[self.initializer])

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    initializer_config[self.initializer](m.weight)
            initializer_config[self.initializer](
                self.predict_layer.weight, 
                **initializer_param_config[self.initializer])
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)
        
            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            
            # predict layers
            predict_weight = torch.cat([self.GMF_model.predict_layer.weight, 
                                        self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device)
            loss = self.criterion(pos_pred, label)

            loss += self.reg_1 * (self.embed_item_GMF(pos_item).weight.norm(p=1))
            loss += self.reg_1 * (self.embed_item_MLP(pos_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item_GMF(pos_item).weight.norm())
            loss += self.reg_2 * (self.embed_item_MLP(pos_item).weight.norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (self.embed_item_GMF(pos_item).weight.norm(p=1) + self.embed_item_GMF(neg_item).weight.norm(p=1))
            loss += self.reg_1 * (self.embed_item_MLP(pos_item).weight.norm(p=1) + self.embed_item_GMF(neg_item).weight.norm(p=1))
            loss += self.reg_2 * (self.embed_item_GMF(pos_item).weight.norm() + self.embed_item_GMF(neg_item).weight.norm())
            loss += self.reg_2 * (self.embed_item_MLP(pos_item).weight.norm() + self.embed_item_GMF(neg_item).weight.norm())
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        loss += self.reg_1 * (self.embed_user_GMF(user).weight.norm(p=1))
        loss += self.reg_1 * (self.embed_user_MLP(user).weight.norm(p=1))
        loss += self.reg_2 * (self.embed_user_GMF(user).weight.norm())
        loss += self.reg_2 * (self.embed_user_MLP(user).weight.norm())

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

            if not self.model == 'MLP':
                embed_user_GMF = self.embed_user_GMF(us).unsqueeze(dim=1) # batch * 1 * factor
                embed_item_GMF = self.embed_item_GMF(cands_ids) # batch * cand_num * factor
                output_GMF = embed_user_GMF * embed_item_GMF # batch * cand_num * factor
            if not self.model == 'GMF':
                embed_user_MLP = self.embed_user_MLP(us).unsqueeze(dim=1) # batch * 1 * factor
                embed_item_MLP = self.embed_item_MLP(cands_ids) # batch * cand_num * factor
                interaction = torch.cat((embed_user_MLP.expand_as(embed_item_MLP), embed_item_MLP), dim=-1) # batch * cand_num * (2 * factor)
                output_MLP = self.MLP_layers(interaction) # batch * cand_num * dim
            
            if self.model == 'GMF':
                concat = output_GMF
            elif self.model == 'MLP':
                concat = output_MLP
            else:
                concat = torch.cat((output_GMF, output_MLP), -1) # batch * cand_num * (dim + factor)
            scores = self.predict_layer(concat).squeeze() # batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids

    def full_rank(self, u):
        u = torch.tensor(u, self.device)

        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(u) # factor
            embed_item_GMF = self.embed_item_GMF.weight # item * factor
            output_GMF = embed_user_GMF * embed_item_GMF  # item * factor
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(u) # factor
            embed_item_MLP = self.embed_item_MLP.weight # item * factor
            interaction = torch.cat((embed_user_MLP.expand_as(embed_item_MLP), embed_item_MLP), dim=-1) # item * (2*factor)
            output_MLP = self.MLP_layers(interaction) # item * dim

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1) # item * (dim + factor)
        scores = self.predict_layer(concat).squeeze() # item
        
        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
