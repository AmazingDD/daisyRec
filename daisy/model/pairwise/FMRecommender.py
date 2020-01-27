import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

class PairFM(nn.Module):
    def __init__(self, num_features, num_factors, batch_norm, drop_prob,
                 epochs=20, lr=0.01, lamda=0.0, gpuid='0', loss_type='BPR', 
                 early_stop=True):
        super(PairFM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.num_features = num_features
        self.num_factors = num_factors
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.epochs = epochs
        self.lr = lr
        self.lamda = lamda
        self.early_stop = early_stop

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))	
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

        self.loss_type = loss_type

    def forward(self, 
                features_i, 
                feature_values_i, 
                features_j,
                feature_values_j):
        pred_i = self._out(features_i, feature_values_i)
        pred_j = self._out(features_j, feature_values_j)

        return pred_i, pred_j
        
    def _out(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_

        return FM.view(-1)

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train() # Enable dropout and batch_norm

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')

            for feat_i, feat_val_i, feat_j, feat_val_j, label in pbar:
                if torch.cuda.is_available():
                    feat_i = feat_i.cuda()
                    feat_j = feat_j.cuda()
                    feat_val_i = feat_val_i.cuda()
                    feat_val_j = feat_val_j.cuda()
                    label = label.cuda()
                else:
                    feat_i = feat_i.cpu()
                    feat_j = feat_j.cpu()
                    feat_val_i = feat_val_i.cpu()
                    feat_val_j = feat_val_j.cpu()
                    label = label.cpu()

                self.zero_grad()

                pred_i, pred_j = self.forward(feat_i, feat_val_i, feat_j, feat_val_j)

                if self.loss_type == 'BPR':
                    loss = -(pred_i - pred_j).sigmoid().log().sum()
                elif self.loss_type == 'HL':
                    loss = torch.clamp(1 - (pred_i - pred_j) * label, min=0).sum()
                else:
                    raise ValueError(f'Invalid loss type: {self.loss_type}')

                loss += self.lamda * self.embeddings.weight.norm()

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

    def predict(self, feat, feat_value):
        pred, _ = self.forward(feat, feat_value, feat, feat_value)

        return pred.cpu()