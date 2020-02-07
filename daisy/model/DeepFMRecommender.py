import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


class DeepFM(nn.Module):
    def __init__(self, 
                 feature_sizes, 
                 embedding_size=4, 
                 hidden_dims=[32, 32],
                 num_classes=1, 
                 dropout=0.5, 
                 epochs=300,
                 lr=1e-4,
                 wd=0.,
                 use_fm=True,
                 use_deep=True,
                 use_cuda=True, 
                 target='regression', # classification
                 verbose=False, 
                 gpuid='0'):
        """
        Initialize a new network
        Inputs: 
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super(DeepFM, self).__init__()

        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims 
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.dtype = torch.long
        self.bias = nn.Parameter(torch.randn(1))

        self.use_deep = use_deep
        self.use_fm = use_fm
        self.use_cuda = use_cuda
        self.verbose = verbose

        self.target = target

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        """
        init fm part
        """
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes]
        )
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )
        """
        init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, f'linear_{i}', nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, f'batchNorm_{i}', nn.BatchNorm1d(all_dims[i]))
            setattr(self, f'dropout_{i}', nn.Dropout(dropout))

    def forward(self, Xi, Xv):
        """
        sum part
        """
        total_sum = self.bias

        """
        fm part
        """
        if self.use_fm:
            fm_first_order_emb_arr = []
            for i, emb in enumerate(self.fm_first_order_embeddings):
                fm_first_order_emb_arr.append(
                    (torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                )
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

            fm_second_order_emb_arr = []
            for i, emb in enumerate(self.fm_second_order_embeddings):
                fm_second_order_emb_arr.append(
                    (torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                )
            fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
            fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
            fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
            fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2

            fm_second_order = 0.5 * (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum)

            total_sum += torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) 
        
        """
        deep part
        """
        if self.use_deep:
            deep_emb = torch.cat(fm_second_order_emb_arr, 1)
            deep_out = deep_emb
            for i in range(1, len(self.hidden_dims) + 1):
                deep_out = getattr(self, f'linear_{i}')(deep_out)
                deep_out = getattr(self, f'batchNorm_{i}')(deep_out)
                deep_out = getattr(self, f'dropout_{i}')(deep_out)

            total_sum += torch.sum(deep_out, 1)        

        return total_sum

    def fit(self, train_loader, val_loader=None, print_each=50):
        if self.use_cuda and torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.wd)
        if self.target == 'classification':
            criterion = nn.BCEWithLogitsLoss()
        elif self.target == 'regression':
            criterion = nn.MSELoss()

        self.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            sum_loss = 0.
            for xi, xv, y in train_loader:
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)
            
                pred = self.forward(xi, xv)
                loss = criterion(pred, y)
                sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elapsed_time = time.time() - start_time
            if self.verbose and epoch % print_each == 0:
                print(f'[Epoch {epoch}] loss={sum_loss:.4f} time cost: {elapsed_time}')
                if val_loader is not None:
                    self._chk_acc(val_loader)

    def predict(self, Xi, Xv):
        """
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        pred = self.forward(Xi, Xv).cpu()

        return pred

    def _chk_acc(self, loader):
        print('Checking accuracy on validation set')
        self.eval()
        
