import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class VAE(nn.Module):
    def __init__(self,
                 rating_mat,
                 q_dims=None,
                 q=0.5,
                 epochs=10,
                 lr=1e-3,
                 reg_1=0.,
                 reg_2=0.,
                 loss_type='CL',
                 gpuid='0',
                 early_stop=True):
        """
        VAE Recommender Class
        Parameters
        ----------
        rating_mat : np.matrix,
        q_dims : List, Q-net dimension list
        q : float, drop out rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(VAE, self).__init__()

        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.loss_type = loss_type
        self.early_stop = early_stop

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        # torch.cuda.set_device(int(gpuid)) # if internal error, try this code instead

        cudnn.benchmark = True

        user_num, item_num = rating_mat.shape
        self.user_num = user_num
        self.item_num = item_num
        self.rating_mat = rating_mat

        p_dims = [200, 600, item_num]
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        
        # Last dimension of q- network is for mean and variance
        tmp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # essay setting only focus on 2 encoder
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(tmp_q_dims[:-1], tmp_q_dims[1:])]
        )
        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )
        self.drop = nn.Dropout(q)
        self._init_weights()

        self.prediction = None

    def _init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]  
                logvar = h[:, self.q_dims[-1]:]
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # core calculation for predicting the real distribution
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        if self.loss_type == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='sum')
        elif self.loss_type == 'SL':
            criterion = nn.MSELoss(reduction='sum')
        else:
            raise ValueError('Invalid loss type')

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for _, ur, mask_ur in pbar:
                if torch.cuda.is_available():
                    ur = ur.cuda()
                    mask_ur = mask_ur.cuda()
                else:
                    ur = ur.cpu()
                    mask_ur = mask_ur.cpu()

                self.zero_grad()
                ur = ur.float()
                mask_ur = mask_ur.float()
                pred, mu, logvar = self.forward(mask_ur)
                # BCE
                # BCE = -torch.mean(torch.sum(F.log_softmax(pred, 1) * ur, -1))
                loss = criterion(pred * mask_ur, ur * mask_ur)
                KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss += KLD

                for layer in self.q_layers:
                    loss += self.reg_1 * layer.weight.norm(p=1)
                    loss += self.reg_2 * layer.weight.norm()
                for layer in self.p_layers:
                    loss += self.reg_1 * layer.weight.norm(p=1)
                    loss += self.reg_2 * layer.weight.norm()

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')
                
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

        x_items = torch.tensor(self.rating_mat).float()
        if torch.cuda.is_available():
            x_items = x_items.cuda()
        else:
            x_items = x_items.cpu()

        # consider there is no enough GPU memory to calculate, we must turn to other methods
        self.prediction = self.forward(x_items)[0]
        self.prediction.clamp_(min=0, max=5)
        self.prediction = self.prediction.cpu().detach().numpy()

        # get row number
        #row_tmp = x_items.size()[0]
        #for idx in range(row_tmp):
        #    tmp = x_items[idx, :].unsqueeze(dim=0)
        #    tmp_pred = self.forward(tmp)[0]
        #    tmp_pred.clamp_(min=0, max=5)
        #    tmp_pred = tmp_pred.cpu().detach().numpy()
        #    if idx == 0:
        #        self.prediction = tmp_pred
        #    else:
        #        self.prediction = np.vstack((self.prediction, tmp_pred))

    def predict(self, u, i):
        return self.prediction[u, i]
