'''
@Author: Yu Di
@Date: 2019-12-05 10:41:31
@LastEditors: Yudi
@LastEditTime: 2019-12-05 14:22:28
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

class BPRMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num=32, 
                 epochs=20, lr=0.01, wd=0.0001, gpuid='0', verbose=True):
        '''
        user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
        '''
        super(BPRMF, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.wd = wd

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.verbose = verbose

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        pred_i = (user * item_i).sum(dim=-1)
        pred_j = (user * item_j).sum(dim=-1)

        return pred_i, pred_j

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        for epoch in range(1, self.epochs + 1):
            self.train()

            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item_i, item_j, _ in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item_i = item_i.cuda()
                    item_j = item_j.cuda()
                else:
                    user = user.cpu()
                    item_i = item_i.cpu()
                    item_j = item_j.cpu()

                self.zero_grad()
                pred_i, pred_j = self.forward(user, item_i, item_j)
                loss = -(pred_i - pred_j).sigmoid().log().sum()

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())

            self.eval()

    def predict(self, u, i):
        u = torch.tensor(u)
        i = torch.tensor(i)

        pred_i, _ = self.forward(u, i, i)

        return pred_i.cpu().item()
