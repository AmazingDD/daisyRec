'''
@Author: Yu Di
@Date: 2019-12-03 15:37:46
@LastEditors: Yudi
@LastEditTime: 2019-12-09 16:39:58
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

class CLMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num=100, lamda=0.0,
                 epochs=20, lr=0.01, wd=0.0001, gpuid='0', verbose=True):
        '''
        user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
        '''
        super(CLMF, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.lr = lr
        self.wd = wd
        self.lamda = lamda
        self.epochs = epochs
        self.verbose = verbose

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        pred = (embed_user * embed_item).sum(dim=-1)

        return pred

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

        for epoch in range(1, self.epochs + 1):
            self.train()

            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item, label in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item = item.cuda()
                    label = label.cuda()
                else:
                    user = user.cpu()
                    item = item.cpu()
                    label = label.cpu()

                self.zero_grad()
                prediction = self.forward(user, item)

                loss = criterion(prediction, label)
                loss += self.lamda * (self.embed_item.weight.norm() +self.embed_user.weight.norm())
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())

            self.eval()
        print('Finish Training Process......')

    def predict(self, u, i):
        pred = self.forward(u, i).cpu()
        
        return pred
