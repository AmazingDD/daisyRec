import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


class PairNFM(nn.Module):
    def __init__(self,
                 user_num, 
                 item_num, 
                 factors, 
                 act_function,
                 num_layers,
                 batch_norm,
                 q, 
                 epochs, 
                 lr, 
                 reg_1=0.,
                 reg_2=0.,
                 loss_type='BPR', 
                 gpuid='0', 
                 early_stop=True):
        """
        Pair-wise NFM Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        act_function : str, activation function for hidden layer
        num_layers : int, number of hidden layers
        batch_norm : bool, whether to normalize a batch of data
        q : float, dropout rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PairNFM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.factors = factors
        self.act_function = act_function
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = q

        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.epochs = epochs
        self.loss_type = loss_type
        self.early_stop = early_stop

        self.embed_user = nn.Embedding(user_num, factors)
        self.embed_item = nn.Embedding(item_num, factors)

        self.u_bias = nn.Embedding(user_num, 1)
        self.i_bias = nn.Embedding(item_num, 1)

        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(factors))
        FM_modules.append(nn.Dropout(self.dropout))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_modules = []
        in_dim = factors
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
        predict_size = factors # layers[-1] if layers else factors

        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.constant_(self.u_bias.weight, 0.0)
        nn.init.constant_(self.i_bias.weight, 0.0)

        # for deep layers
        if self.num_layers > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, u, i, j):
        user = self.embed_user(u)
        item_i = self.embed_item(i)
        item_j = self.embed_item(j)

        # inner product part
        pred_i = (user * item_i)
        pred_j = (user * item_j)
        pred_i = self.FM_layers(pred_i)
        pred_j = self.FM_layers(pred_j)

        if self.num_layers:
            pred_i = self.deep_layers(pred_i)
            pred_j = self.deep_layers(pred_j)

        pred_i += self.u_bias(u) + self.i_bias(i) + self.bias_
        pred_j += self.u_bias(u) + self.i_bias(j) + self.bias_

        pred_i = self.prediction(pred_i)
        pred_j = self.prediction(pred_j)

        return pred_i.view(-1), pred_j.view(-1)

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.Adagrad(self.parameters(), lr=self.lr, initial_accumulator_value=1e-8)

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item_i, item_j, label in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item_i = item_i.cuda()
                    item_j = item_j.cuda()
                    label = label.cuda()
                else:
                    user = user.cpu()
                    item_i = item_i.cpu()
                    item_j = item_j.cpu()
                    label = label.cpu()

                self.zero_grad()
                pred_i, pred_j = self.forward(user, item_i, item_j)

                if self.loss_type == 'BPR':
                    loss = -(pred_i - pred_j).sigmoid().log().sum()
                elif self.loss_type == 'HL':
                    loss = torch.clamp(1 - (pred_i - pred_j) * label, min=0).sum()
                elif self.loss_type == 'TL':
                    loss = (pred_j - pred_i).sigmoid().mean() + pred_j.pow(2).sigmoid().mean()
                else:
                    raise ValueError(f'Invalid loss type: {self.loss_type}')

                loss += self.reg_1 * (self.embed_item.weight.norm(p=1) + self.embed_user.weight.norm(p=1))
                loss += self.reg_2 * (self.embed_item.weight.norm() + self.embed_user.weight.norm())

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()
        
            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def predict(self, u, i):
        pred_i, _ = self.forward(u, i, i)

        return pred_i.cpu()