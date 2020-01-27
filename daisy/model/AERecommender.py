import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

class AE(nn.Module):
    def __init__(self, user_num, item_num, hidden_neuron, epochs, lr, lamda, gpuid='0'):
        super(AE, self).__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.lr = lr
        self.lamda = lamda
        self.epochs = epochs

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.Encoder = nn.Linear(item_num, hidden_neuron)
        self.Decoder = nn.Linear(hidden_neuron, item_num)
        self.activation = nn.Sigmoid()

        nn.init.normal_(self.Encoder.weight, std=0.03)
        nn.init.normal_(self.Decoder.weight, std=0.03)

    def forward(self, x):
        x = self.activation(self.Encoder(x))
        x = self.Decoder(x)

        return x

    def fit(self, train_loader, r_all):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.train()

            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for ser, mask_ser in pbar:
                if torch.cuda.is_available():
                    ser = ser.cuda()
                    mask_ser = mask_ser.cuda()
                else:
                    ser = ser.cpu()
                    mask_ser = mask_ser.cpu()

                self.zero_grad()
                
                ser = ser.float()
                mask_ser = mask_ser.float()

                pred = self.forward(ser)

                loss = criterion(pred * mask_ser, ser * mask_ser)
                loss += self.lamda * (self.Encoder.weight.norm() + self.Decoder.weight.norm())
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
            self.eval()

        self.prediction = self.forward(torch.tensor(r_all).float())
        self.prediction.clamp_(min=0, max=5)
        self.prediction = self.prediction.detach().numpy()

    
    def predict(self, u, i):
        return self.prediction[u, i]
