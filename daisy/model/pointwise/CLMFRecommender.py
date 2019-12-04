'''
@Author: Yu Di
@Date: 2019-12-03 15:37:46
@LastEditors: Yudi
@LastEditTime: 2019-12-03 22:59:51
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
    def __init__(self, user_num, item_num, factor_num=100, 
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
        self.epochs = epochs
        self.verbose = verbose

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.predict_layer = nn.Linear(factor_num, 1)

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        pred = self.predict_layer(embed_user * embed_item)

        return pred.view(-1)

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_function = nn.BCEWithLogitsLoss()

        for epoch in range(1, self.epochs + 1):
            self.train()

            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item, label in pbar:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item = item.cuda()
                else:
                    user = user.cpu()
                    item = item.cpu()

                self.zero_grad()
                prediction = self.forward(user, item)
                loss = loss_function(prediction, label)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())

            self.eval()
        print('Finish Training Process......')

    def predict(self, u, i):
        u = torch.tensor(u)
        i = torch.tensor(i)
        return self.forward(u, i).cpu().item()
