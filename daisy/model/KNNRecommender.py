import os
import heapq
import numpy as np
from six import iteritems
from collections import defaultdict

from daisy.model.similarities import cosine, jaccard, pearson

class SymmetricAlgo(object):
    def __init__(self, user_num, item_num, **kwargs):
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based']=True

        self.user_num = user_num
        self.item_num = item_num

    def fit(self, train_set):
        self.ur, self.ir = defaultdict(list), defaultdict(list)
        for _, row in train_set.iterrows():
            self.ur[int(row['user'])].append((int(row['item']), row['rating']))
            self.ir[int(row['item'])].append((int(row['user']), row['rating']))

        ub = self.sim_options['user_based']
        self.n_x = self.user_num if ub else self.item_num
        self.n_y = self.item_num if ub else self.user_num
        self.xr = self.ur if ub else self.ir
        self.yr = self.ir if ub else self.ur

        return self
    
    def switch(self, u_stuff, i_stuff):
        '''Return x_stuff and y_stuff depending on the user_based field.'''
        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

    def compute_similarities(self):
        construction_func = {'cosine': cosine,
                             'pearson': pearson,
                             'jaccard': jaccard}
        if self.sim_options['user_based']:
            n_x, yr, xr = self.user_num, self.ir, self.ur
        else: 
            n_x, yr, xr = self.item_num, self.ur, self.ir

        min_support = self.sim_options.get('min_support', 1)
        args = [n_x, yr, xr, min_support]

        name = self.sim_options.get('name', 'cosine').lower()
        
        try:
            print('Computing the {0} similarity matrix...'.format(name))
            sim = construction_func[name](*args)
            print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError(f'Wrong sim name {name}. Allowed values are ' + ', '.join(construction_func.keys()) + '.')

class KNNWithMeans(SymmetricAlgo):
    ''' KNN with means '''
    def __init__(self, user_num, item_num, k=40, min_k=1, sim_options={}, 
                 verbose=True, tune_or_not=False, 
                 serial='ml-100k-origin-loo-0-cosine', **kwargs):
        if 'user_based' not in sim_options:
            sim_options['user_based']=True

        self.sim_base = 'user' if sim_options['user_based'] else 'item'

        SymmetricAlgo.__init__(self, user_num, item_num, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

        # only used when tuning
        self.tune_or_not = tune_or_not
        self.serial = serial

        if not os.path.exists('./tmp/sim_matrix/'):
            os.makedirs('./tmp/sim_matrix/')

    def fit(self, train_set):
        SymmetricAlgo.fit(self, train_set)
        if self.tune_or_not:
            # if you want to tune
            sim_file_path = f'./tmp/sim_matrix/{self.sim_base}_sim_mat_{self.serial}.npy'
            if os.path.exists(sim_file_path):
                self.sim = np.load(sim_file_path)
                print(f'Load similarity matrix, serial: {self.serial}')
            else:
                sim_mat = self.compute_similarities()
                self.sim = sim_mat
                np.save(sim_file_path, sim_mat)
        else:
            # if you just run the result
            self.sim = self.compute_similarities()

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def predict(self, u, i):
        if u >= self.user_num or i >= self.item_num:
            raise ValueError('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        # details = {'actual_k': actual_k}
        # return est, details

        return est
