'''
@Author: Yu Di
@Date: 2019-12-10 16:14:00
@LastEditors: Yudi
@LastEditTime: 2019-12-11 09:55:20
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

class BPRSLiM(nn.Module):
    def __init__(self, data, user_num, item_num, epochs,
                 lr=0.01, beta=0.0, lamda=0.0, gpuid='0'):
        super(BPRSLiM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.user_num = user_num
        self.item_num = item_num

        A = np.zeros((user_num, item_num))
        for _, row in data.iterrows():
            user, item = int(row['user']), int(row['item'])
            A[user, item] = 1
        self.A = A  # user-item matrix

        self.W = nn.Embedding(item_num, item_num)
        nn.init.normal_(self.W.weight, std=0.01)
        # force weight >=0 and diag(W) = 0
        self.W.weight.data.clamp_(0)
        self.W.weight.data.copy_(self.W.weight.data * (1 - torch.eye(self.item_num,self.item_num)))

        self.epochs = epochs
        self.lr = lr
        self.beta = beta / 2  # Frobinious regularization
        self.lamda = lamda # lasso regularization

    def forward(self, user, item_i, item_j):
        tensor_A = torch.from_numpy(self.A).to(torch.float32)
        ru = tensor_A[user]
        wi = self.W(item_i)
        wj = self.W(item_j)

        pred_i = (ru * wi).sum(dim=-1)
        pred_j = (ru * wj).sum(dim=-1)

        return pred_i, pred_j

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()
        
        optimizer = optim.SGD(self.parameters(), self.lr)
        
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
                loss += self.beta * self.W.weight.norm() + self.lamda * self.W.weight.norm(p=1)

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
            
            self.eval()

            # reset W element >= 0 and diag(W) = 0
            self.W.weight.data.clamp_(0)
            self.W.weight.data.copy_(self.W.weight.data * (1 - torch.eye(self.item_num,self.item_num))) 

        print('Finish Training Process......')

        # get prediction rating matrix
        self.A_tilde = self.A.dot(self.W.weight.data.numpy())

    def predict(self, u, i):
        return self.A_tilde[u, i]