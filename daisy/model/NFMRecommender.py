import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

class NFM(nn.Module):
    def __init__(self, num_features, num_factors, act_function, 
                 layers, batch_norm, drop_prob, epochs, lr, lamda,
                 opt_func, loss_type, pretrain_FM=None):
        super(NFM, self).__init__()
        """
		num_features: number of features,
		num_factors: number of hidden factors,
		act_function: activation function for MLP layer,
		layers: list of dimension of deep layers,
		batch_norm: bool type, whether to use batch norm or not,
		drop_prob: list of the dropout rate for FM and MLP,
		pretrain_FM: the pre-trained FM weights.
		"""
        self.num_features = num_factors
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.pretrain_FM = pretrain_FM

        self.lr = lr
        self.opt_function = opt_func
        self.loss_type = loss_type
        self.epochs = epochs

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))	
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        
        self.deep_layers = nn.Sequential(*MLP_module)
        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight_()

    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """
        if self.pretrain_FM:
            self.embeddings.weight.data.copy_(self.pretrain_FM.embeddings.weight)
            self.biases.weight.data.copy_(self.pretrain_FM.biases.weight)
            self.bias_.data.copy_(self.pretrain_FM.bias_)

        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        FM = self.prediction(FM)

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

        if self.opt_function == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.lr, initial_accumulator_value=1e-8)
        elif self.opt_function == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt_function == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.opt_function == 'Momentum':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.95)

        if self.loss_type == 'SL':
            criterion = nn.MSELoss(reduction='sum')
        elif self.loss_type == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')

            for features, feature_values, labels in pbar:
                if torch.cuda.is_available():
                    features = features.cuda()
                    feature_values = feature_values.cuda()
                    labels = labels.cuda()
                else:
                    features = features.cpu()
                    feature_values = feature_values.cpu()
                    labels = labels.cpu()

                self.zero_grad()
                prediction = self.forward(features, feature_values)
                loss = criterion(prediction, labels) 
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
    
    def predict(self, features, feature_values):
        pred = self.forward(features, feature_values)

        return pred.cpu()
