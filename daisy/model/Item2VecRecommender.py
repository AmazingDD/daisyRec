import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class Bundler(nn.Module):
    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class ItemEmb(Bundler):
    def __init__(self, item_num=20000, factors=100, padding_idx=0):
        super(ItemEmb, self).__init__()
        self.item_num = item_num # item_num
        self.factors = factors
        self.ivectors = nn.Embedding(self.item_num, self.factors, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.item_num, self.factors, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.factors),
                                                       torch.FloatTensor(
                                                           self.item_num - 1, self.factors).uniform_(
                                                           -0.5 / self.factors, 0.5 / self.factors)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.factors),
                                                       torch.FloatTensor(self.item_num - 1,
                                                                         self.factors).uniform_(-0.5 / self.factors,
                                                                                                0.5 / self.factors)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = torch.LongTensor(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = torch.LongTensor(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class Item2Vec(nn.Module):
    def __init__(self, 
                 item2idx,
                 item_num=20000, 
                 factors=100, 
                 epochs=20,
                 n_negs=20,
                 padding_idx=0, 
                 weights=None, 
                 use_cuda=False, 
                 early_stop=True):
        """
        Item2Vec Recommender Class with SGNS
        Parameters
        ----------
        item2idx : Dict, {item : index code} relation mapping
        item_num : int, total item number
        factors : int, latent factor number
        epochs : int, training epoch number
        n_negs : int, number of negative samples
        padding_idx : int, pads the output with the embedding vector at padding_idx (initialized to zeros)
        weights : weight value to use
        use_cuda : bool, whether to use CUDA enviroment
        early_stop : bool, whether to activate early stop mechanism
        """
        super(Item2Vec, self).__init__()
        self.item2idx = item2idx
        self.embedding = ItemEmb(item_num, factors, padding_idx)
        self.item_num = item_num
        self.factors = factors
        self.epochs = epochs
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = torch.FloatTensor(wf)

        self.use_cuda = use_cuda
        self.early_stop = early_stop

        self.user_vec_dict = {}
        self.item2vec = {}

    def forward(self, iitem, oitems):
        batch_size = iitem.size()[0]
        context_size = oitems.size()[1]
        if self.weights is not None:
            nitems = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, 
                                       replacement=True).view(batch_size, -1)
        else:
            nitems = torch.FloatTensor(batch_size,
                                       context_size * self.n_negs).uniform_(0, self.item_num - 1).long()
        ivectors = self.embedding.forward_i(iitem).unsqueeze(2)
        ovectors = self.embedding.forward_o(oitems)
        nvectors = self.embedding.forward_o(nitems).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, 
                                                                             self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

    def fit(self, train_loader):
        item2idx = self.item2idx
        if self.use_cuda and torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()    

        optimizer = optim.Adam(self.parameters())
        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch}]')
            for iitem, oitems in pbar:
                loss = self.forward(iitem, oitems)

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()
            
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

        idx2vec = self.embedding.ivectors.weight.data.cpu().numpy()

        self.item2vec = {k: idx2vec[v, :] for k, v in item2idx.items()}

    def build_user_vec(self, ur):
        for u in ur.keys():
            self.user_vec_dict[u] = np.array([self.item2vec[i] for i in ur[u]]).mean(axis=0)

    def predict(self, u, i):
        if u in self.user_vec_dict.keys():
            user_vec = self.user_vec_dict[u]
        else:
            return 0.
        item_vec = self.item2vec[i]

        return self._cos_sim(user_vec, item_vec)

    def _cos_sim(self, a, b):
        numerator = np.multiply(a, b).sum()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator
