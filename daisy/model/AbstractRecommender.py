import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from daisy.utils.config import initializer_param_config, initializer_config
from daisy.utils.loss import BPRLoss, TOP1Loss, HingeLoss


class AbstractRecommender(nn.Module):
    def __init__(self):
        super(AbstractRecommender, self).__init__()
        self.optimizer = None
        self.initializer = None
        self.loss_type = None
        self.lr = 0.01
        self.logger = None

    def calc_loss(self, batch):
        raise NotImplementedError

    def fit(self, train_loader):
        raise NotImplementedError

    def rank(self, test_loader):
        raise NotImplementedError

    def full_rank(self, u):
        raise NotImplementedError

    def predict(self, u, i):
        raise NotImplementedError

    def _build_optimizer(self, **kwargs):
        params = self.parameters()
        learner = kwargs.pop('optimizer', self.optimizer)
        learning_rate = kwargs.pop('lr', self.lr)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            self.logger.info('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)

        return optimizer

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            initializer_config[self.initializer](m.weight, **initializer_param_config[self.initializer])
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            initializer_config[self.initializer](m.weight, **initializer_param_config[self.initializer])
        else:
            pass

    def _build_criterion(self, loss_type):
        if loss_type.upper() == 'CL':
            criterion = nn.BCEWithLogitsLoss(reduction='sum')
        elif loss_type.upper() == 'SL':
            criterion = nn.MSELoss(reduction='sum')
        elif loss_type.upper() == 'BPR':
            criterion = BPRLoss()
        elif loss_type.upper() == 'HL':
            criterion = HingeLoss()
        elif loss_type.upper() == 'TL':
            criterion = TOP1Loss()
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}...')

        return criterion

class GeneralRecommender(AbstractRecommender):
    def __init__(self, config):
        super(GeneralRecommender, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.logger = config['logger']

    def fit(self, train_loader):
        self.to(self.device)
        optimizer = self._build_optimizer(optimizer=self.optimizer, lr=self.lr)
        self.criterion = self._build_criterion(self.loss_type)

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for batch in pbar:
                self.zero_grad()
                loss = self.calc_loss(batch)

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                current_loss += loss.item()
            pbar.set_postfix(loss=current_loss)

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                self.logger.info('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

class AERecommender(GeneralRecommender):
    def __init__(self, config):
        super(AERecommender, self).__init__(config)
        self.user_num = None 
        self.item_num = None
        self.history_user_id, self.history_item_id = None, None
        self.history_user_value, self.history_item_value = None, None

    def get_user_rating_matrix(self, user):
        '''
        just convert the raw rating matrix to a much smaller matrix for calculation,
        the row index will be the new id for uid, but col index will still remain the old iid
        '''
        col_indices = self.history_item_id[user].flatten() # batch * max_inter_by_user -> (batch * max_inter_by_user)
        row_indices = torch.arange(user.shape[0]).to(self.device).repeat_interleave(
            self.history_item_id.shape[1], dim=0) # batch -> (batch * max_inter_by_user)
        rating_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.item_num) # batch * item_num
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        
        return rating_matrix

    def get_item_rating_matrix(self, item):
        col_indices = self.history_user_id[item].flatten()
        row_indices = torch.arange(item.shape[0]).to(self.device).repeat_interleave(
            self.history_user_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1).to(self.device).repeat(item.shape[0], self.user_num)
        rating_matrix.index_put_((row_indices, col_indices), self.history_user_value[item].flatten())
        
        return rating_matrix
