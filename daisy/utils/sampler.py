import numpy as np

class AbstractSampler(object):
    def __init__(self, config):
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.item_num = config['item_num']
        self.ur = config['train_ur']

    def sampling(self):
        raise NotImplementedError

class BasicNegtiveSampler(AbstractSampler):
    def __init__(self, df, config):
        """
        negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
        Parameters
        ----------
        df : pd.DataFrame, the raw <u, i, r> dataframe
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, No. of nagative sampling per sample, default is 4
        sample_method : str, sampling method, default is 'uniform',
                        'uniform' discrete uniform sampling
                        'high-pop' sample items with high popularity as priority
                        'low-pop' sample items with low popularity as prority
        sample_ratio : float, scope [0, 1], it determines the ratio that the other sample method except 'uniform' occupied, default is 0
        """
        super(BasicNegtiveSampler, self).__init__(config)
        self.user_num = config['user_num']
        self.num_ng = config['num_ng']
        self.inter_name = config['INTER_NAME']
        self.sample_method = config['sample_method']
        self.sample_ratio = config['sample_ratio']
        self.loss_type = config['loss_type'].upper()

        assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None
        
        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(self.item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(self.item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def sampling(self):
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

        js = np.zeros((self.user_num, self.num_ng), dtype=np.int32)
        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num

            for u in range(self.user_num):
                past_inter = list(self.ur[u])

                uni_negs = np.random.choice(
                    np.setdiff1d(np.arange(self.item_num), past_inter), 
                    size=uniform_num
                )
                other_negs = np.random.choice(
                    np.arange(self.item_num),
                    size=other_num,
                    p=self.pop_prob
                )
                js[u] = np.concatenate((uni_negs, other_negs), axis=None)

        else:
            # all negative samples are sampled by uniform distribution
            for u in range(self.user_num):
                past_inter = list(self.ur[u])
                js[u] = np.random.choice(
                    np.setdiff1d(np.arange(self.item_num), past_inter), 
                    size=self.num_ng
                )

        self.df['neg_set'] = self.df[self.uid_name].agg(lambda u: js[u])

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name, self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')    
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name, 'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError

class SkipGramNegativeSampler(AbstractSampler):
    def __init__(self, df, config, discard=False):
        '''
        skip-gram negative sampling class for <target_i, context_i, label>

        Parameters
        ----------
        df : pd.DataFrame
            training set
        rho : float, optional
            threshold to discard word in a sequence, by default 1e-5
        context_window: int, context range around target
        train_ur: dict, ground truth for each user in train set
        item_num: int, the number of items
        '''    
        super(SkipGramNegativeSampler, self).__init__(config)    
        self.context_window = config['context_window']

        word_frequecy = df[self.iid_name].value_counts()
        prob_discard = 1 - np.sqrt(config['rho'] / word_frequecy)

        if discard:
            rnd_p = np.random.uniform(low=0., high=1., size=len(df))
            discard_p_per_item = df[self.iid_name].map(prob_discard).values
            df = df[rnd_p >= discard_p_per_item]

        self.train_seqs = self._build_seqs(df)

    def sampling(self):
        sgns_samples = []

        for u, seq in self.train_seqs.iteritems():
            past_inter = list(self.ur[u])
            cands = np.setdiff1d(np.arange(self.item_num), past_inter)

            for i in range(len(seq)):
                target = seq[i]
                # generate positive sample
                context_list = []
                j = i - self.context_window
                while j <= i + self.context_window and j < len(seq):
                    if j >= 0 and j != i:
                        context_list.append(seq[j])
                        sgns_samples.append([target, seq[j], 1])
                    j += 1
                # generate negative sample
                num_ng = len(context_list)
                for neg_item in np.random.choice(cands, size=num_ng):
                    sgns_samples.append([target, neg_item, 0])
        
        return np.array(sgns_samples)

    def _build_seqs(self, df):
        train_seqs = df.groupby(self.uid_name)[self.iid_name].agg(list)

        return train_seqs
