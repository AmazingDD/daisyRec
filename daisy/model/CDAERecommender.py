import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


class CDAE(nn.Module):
    def __init__(self, 
                 rating_mat,
                 factors=20, 
                 act_activation='relu', 
                 out_activation='sigmoid', 
                 epochs=10,
                 lr=0.001,
                 q=0.5, 
                 reg_1=0., 
                 reg_2=0.01,
                 loss_type='CL',
                 gpuid='0',
                 early_stop=True):
        """
        CDAE Recommender Class
        Parameters
        ----------
        rating_mat : np.matrix, rating matrix
        factors : int, latent factor number
        act_activation : str, activation function for hidden layer, default is 'relu'
        out_activation : str, activation function for output layer, default is 'sigmoid'
        epochs : int, number of training epochs
        lr : float, learning rate
        q : float, drop out rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(CDAE, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        user_num, item_num = rating_mat.shape
        self.user_num = user_num 
        self.item_num = item_num
        self.rating_mat = rating_mat

        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.early_stop = early_stop
        self.epochs = epochs
        self.lr = lr
        self.loss_type=loss_type
        self.h_item = nn.Linear(item_num, factors)

        if act_activation == 'sigmoid':
            self.h_act = nn.Sigmoid()
        elif act_activation == 'relu':
            self.h_act = nn.ReLU()
        elif act_activation == 'tanh':
            self.h_act = nn.Tanh()
        else:
            raise ValueError('Invalid hidden layer activation function')

        if out_activation == 'sigmoid':
            self.o_act = nn.Sigmoid()
        elif out_activation == 'relu':
            self.o_act = nn.ReLU()
        else:
            raise ValueError('Invalid output layer activation function')

        self.dropout = nn.Dropout(p=q)

        # dtype should be int to connect to Embedding layer
        self.h_user = nn.Embedding(user_num, factors)

        self.out = nn.Linear(factors, item_num)

        self.prediction = None  # this is used for storing prediction result

    def forward(self, x_user, x_items):
        h_i = self.dropout(x_items)
        h_i = self.h_item(h_i)

        h_u = self.h_user(x_user)

        h = torch.add(h_u, h_i)
        out = self.out(h)

        return out

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

            for user, ur, mask_ur in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    ur = ur.cuda()
                    mask_ur = mask_ur.cuda()
                else:
                    user = user.cpu()
                    ur = ur.cpu()
                    mask_ur = mask_ur.cpu()

                self.zero_grad()
                ur = ur.float()
                mask_ur = mask_ur.float()

                pred = self.forward(user, mask_ur)

                loss = criterion(pred * mask_ur, ur * mask_ur)  # only concern loss with known interaction
                # l1-regularization
                loss += self.reg_1 * (self.h_user.weight.norm(p=1) + self.h_item.weight.norm(p=1))
                # l2-regularization
                loss += self.reg_2 * (self.h_user.weight.norm() + self.h_item.weight.norm())

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

        x_user = torch.tensor(list(range(self.user_num)))
        x_items = torch.tensor(self.rating_mat).float()
        prediction = self.forward(x_user, x_items)
        prediction.clamp_(min=0, max=5)
        self.prediction = prediction.detach().numpy()

    def predict(self, u, i):
        return self.prediction[u, i]
