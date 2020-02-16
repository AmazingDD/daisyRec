import os
import gc
import re
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch.utils.data as data

from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit

def load_rate(src='ml-100k', prepro='origin', binary=True, pos_threshold=None):
    # which dataset will use
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')

    elif src == 'ml-1m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
        # only consider rating >=4 for data density
        df = df.query('rating >= 4').reset_index(drop=True).copy()

    elif src == 'ml-10m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df = df.query('rating >= 4').reset_index(drop=True).copy()

    elif src == 'ml-20m':
        df = pd.read_csv(f'./data/{src}/ratings.csv')
        df.rename(columns={'userId':'user', 'movieId':'item'}, inplace=True)
        df = df.query('rating >= 4').reset_index(drop=True)

    elif src == 'netflix':
        cnt = 0
        tmp_file = open(f'./data/{src}/training_data.csv', 'w')
        tmp_file.write('user,item,rating,timestamp' + '\n')
        for f in os.listdir(f'./data/{src}/training_set/'):
            cnt += 1
            if cnt % 5000 == 0:
                print(f'Finish Process {cnt} file......')
            txt_file = open(f'./data/{src}/training_set/{f}', 'r')
            contents = txt_file.readlines()
            item = contents[0].strip().split(':')[0]
            for val in contents[1:]:
                user, rating, timestamp = val.strip().split(',')
                tmp_file.write(','.join([user, item, rating, timestamp]) + '\n')
            txt_file.close()

        tmp_file.close()

        df = pd.read_csv(f'./data/{src}/training_data.csv')
        df['rating'] = df.rating.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    elif src == 'lastfm':
        # user_artists.dat
        df = pd.read_csv(f'./data/{src}/user_artists.dat', sep='\t')
        df.rename(columns={'userID':'user', 'artistID':'item', 'weight':'rating'}, inplace=True)
        # treat weight as interaction, as 1
        df['rating'] = 1.0
        # fake timestamp column
        df['timestamp'] = 1

    elif src == 'bx':
        df = pd.read_csv(f'./data/{src}/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
        df.rename(columns={'User-ID': 'user', 'ISBN': 'item', 'Book-Rating': 'rating'}, inplace=True)
        # fake timestamp column
        df['timestamp'] = 1

    elif src == 'pinterest':
        pass

    elif src == 'amazon-cloth':
        df = pd.read_csv(f'./data/{src}/ratings_Clothing_Shoes_and_Jewelry.csv', 
                        names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'amazon-electronic':
        df = pd.read_csv(f'./data/{src}/ratings_Electronics.csv', 
                        names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'amazon-book':
        df = pd.read_csv(f'./data/{src}/ratings_Books.csv', 
                        names=['user', 'item', 'rating', 'timestamp'], low_memory=False)
        df = df[df['timestamp'].str.isnumeric()].copy()
        df['timestamp'] = df['timestamp'].astype(int)

    elif src == 'amazon-music':
        df = pd.read_csv(f'./data/{src}/ratings_Digital_Music.csv', 
                        names=['user', 'item', 'rating', 'timestamp'])

    elif src == 'epinions':
        d = sio.loadmat(f'./data/{src}/rating_with_timestamp.mat')
        prime = []
        for val in d['rating_with_timestamp']:
            user, item, rating, timestamp = val[0], val[1], val[3], val[5]
            prime.append([user, item, rating, timestamp])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        del prime
        gc.collect()

    elif src == 'yelp':
        json_file_path = f'./data/{src}/yelp_academic_dataset_review.json'
        prime = []
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            val = json.loads(line)
            prime.append([val['user_id'], val['business_id'], val['stars'], val['date']])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df.timestamp)
        del prime
        gc.collect()

    elif src == 'citeulike':
        user = 0
        dt = []
        for line in open(f'./data/{src}/users.dat', 'r'):
            val = line.split()
            for item in val:
                dt.append([user, item])
            user += 1
        df = pd.DataFrame(dt, columns=['user', 'item'])
        # fake timestamp column
        df['timestamp'] = 1

    else:
        raise ValueError('Invalid Dataset Error')

    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rating >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rating'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        df = df.query(f'cnt_item >= {core_num} and cnt_user >= {core_num}').reset_index(drop=True).copy()
        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore (N is int number) expected')

    # encoding user_id and item_id
    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    print(f'Finish loading [{src}]-[{prepro}] dataset')

    return df, user_num, item_num

def negative_sampling(user_num, item_num, df, num_ng=4, neg_label_val=0., sample_method='uniform'):
    """
    :param user_num: # of users
    :param item_num: # of items
    :param df: dataframe for sampling
    :param num_ng: # of nagative sampling per sample
    :param neg_label_val: target value for negative samples
    :param sample_method: 'uniform' discrete uniform 
                          'item-desc' descending item popularity, high popularity means high probability to choose
                          'item-ascd' ascending item popularity, low popularity means high probability to choose
    """
    assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
    neg_sample_pool = list(range(item_num))
    if sample_method != 'uniform':
        popularity_item_list = df['item'].value_counts().index.tolist()
        if sample_method == 'item-desc':
            neg_sample_pool = popularity_item_list
        elif sample_method == 'item-ascd':
            neg_sample_pool = popularity_item_list[::-1]

    pair_pos = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for _, row in df.iterrows():
        pair_pos[int(row['user']), int(row['item'])] = 1.0

    neg_df = []
    for _, row in df.iterrows():
        u = int(row['user'])
        i = int(row['item'])
        r = row['rating']
        neg_df.append([u, i, r, 1])
        for _ in range(num_ng):
            if sample_method == 'uniform':
                j = np.random.randint(item_num)
                while (u, j) in pair_pos:
                    j = np.random.randint(item_num)
            # since item-desc, item-ascd both use neg_sample_pool to sample negative item
            elif sample_method in ['item-desc', 'item-ascd']:
                idx = 0
                j = int(neg_sample_pool[idx])
                while (u, j) in pair_pos:
                    idx += 1
                    j = int(neg_sample_pool[idx])

            j = int(j)
            neg_df.append([u, j, neg_label_val, 1])

    neg_df = pd.DataFrame(neg_df, columns=['user', 'item', 'rating', 'timestamp'])
    print('Finish negative sampling......')

    return neg_df

def split_test(df, test_method='fo', test_size=.2):
    """
    :param df: raw data waiting for test set splitting
    :param test_method: way to split test set
                        'fo': split by ratio
                        'tfo': split by ratio with timesstamp
                        'tloo': leave one out with timestamp
                        'loo': leave one out
                        TODO 'ufo': split by ratio in user level
    """
    if test_method == 'ufo':
        driver_ids = df['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=2020)
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    if test_method == 'tfo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(df) * (1 - test_size)))
        train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()

    elif test_method == 'fo':
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=2019)

    elif test_method == 'tloo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    elif test_method == 'loo':
        # # slow method
        # test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        # test_key = test_set[['user', 'item']].copy()
        # train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()

        # # quick method
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()

    else:
        raise ValueError('Invalid data_split value, expect: loo, fo, tloo, tfo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)

    return train_set, test_set

def split_validation(train_set, val_method='fo', fold_num=1, val_size=.1):
    """
    Parameter
    ---------
    :param train_set: train set waiting for split validation
    :param val_method: way to split validation
                       'cv': combine with fold_num => fold_num-CV
                       'fo': combine with fold_num & val_size => fold_num-Split by ratio(9:1)
                       'tfo': Split by ratio with timestamp, combine with val_size => 1-Split by ratio(9:1)
                       'tloo': Leave one out with timestamp => 1-Leave one out
                       'loo': combine with fold_num => fold_num-Leave one out
                       'ufo': split by ratio in user level with K-fold
    """
    if val_method in ['tloo', 'tfo']:
        cnt = 1
    elif val_method in ['cv', 'loo', 'fo', 'ufo']:
        cnt = fold_num
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')
    
    train_set_list, val_set_list = [], []
    if val_method == 'ufo':
        driver_ids = train_set['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=fold_num, test_size=val_size, random_state=2020)
        for train_idx, val_idx in gss.split(train_set, groups=driver_indices):
            train_set_list.append(train_set.loc[train_idx, :])
            val_set_list.append(train_set.loc[val_idx, :])
    if val_method == 'cv':
        kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train_set):
            train_set_list.append(train_set.loc[train_index, :])
            val_set_list.append(train_set.loc[val_index, :])
    if val_method == 'fo':
        for _ in range(fold_num):
            train, validation = train_test_split(train_set, test_size=val_size)
            train_set_list.append(train)
            val_set_list.append(validation)
    elif val_method == 'tfo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(train_set) * (1 - val_size)))
        train_set_list.append(train_set.iloc[:split_idx, :])
        val_set_list.append(train_set.iloc[split_idx:, :])
    elif val_method == 'loo':
        for _ in range(fold_num):
            # val_set = train_set.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            # val_key = val_set[['user', 'item']].copy()
            # sub_train_set = train_set.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(val_key)).reset_index().copy()

            val_index = train_set.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
            val_set = train_set.loc[val_index, :].reset_index(drop=True).copy()
            sub_train_set = train_set[~train_set.index.isin(val_index)].reset_index(drop=True).copy()

            train_set_list.append(sub_train_set)
            val_set_list.append(val_set)
    elif val_method == 'tloo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        train_set['rank_latest'] = train_set.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        new_train_set = train_set[train_set['rank_latest'] > 1].copy()
        val_set = train_set[train_set['rank_latest'] == 1].copy()
        del new_train_set['rank_latest'], val_set['rank_latest']

        train_set_list.append(new_train_set)
        val_set_list.append(val_set)

    return train_set_list, val_set_list, cnt

def get_ur(df):
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur

def get_ir(df):
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir

def build_feat_idx_dict(df:pd.DataFrame, 
                        cat_cols:list=['user', 'item'], 
                        num_cols:list=[]):
    feat_idx_dict = {}
    idx = 0
    for col in cat_cols:
        feat_idx_dict[col] = idx
        idx = idx + df[col].max() + 1
    for col in num_cols:
        feat_idx_dict[col] = idx
        idx += 1
    print('Finish build feature index dictionary......')

    cnt = 0
    for col in cat_cols:
        for _ in df[col].unique():
            cnt += 1
    for col in num_cols:
        cnt += 1
    print(f'Number of features: {cnt}')

    return feat_idx_dict, cnt

class PointMFData(data.Dataset):
    def __init__(self, sampled_df):
        super(PointMFData, self).__init__()
        self.features_fill = []
        self.labels_fill = []
        for _, row in sampled_df.iterrows():
            self.features_fill.append([int(row['user']), int(row['item'])])
            self.labels_fill.append(row['rating'])
        self.labels_fill = np.array(self.labels_fill, dtype=np.float32)

    def __len__(self):
        return len(self.labels_fill)

    def __getitem__(self, idx):
        features = self.features_fill
        labels = self.labels_fill

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]

        return user, item, label

class PointFMData(data.Dataset):
    def __init__(self, sampled_df, feat_idx_dict, 
                 cat_cols, num_cols, loss_type='square_loss'):
        super(PointFMData, self).__init__()

        self.labels = []
        self.features = []
        self.feature_values = []

        assert loss_type in ['square_loss', 'log_loss'], 'Invalid loss type'
        for _, row in sampled_df.iterrows():
            feat, feat_value = [], []
            for col in cat_cols:
                feat.append(feat_idx_dict[col] + row[col])
                feat_value.append(1)
            for col in num_cols:
                feat.append(feat_idx_dict[col])
                feat_value.append(row[col])
            self.features.append(np.array(feat, dtype=np.int64))
            self.feature_values.append(np.array(feat_value, dtype=np.float32))

            if loss_type == 'square_loss':
                self.labels.append(np.float32(row['rating']))
            else: # log_loss
                label = 1 if float(row['rating']) > 0 else 0
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        features = self.features[idx]
        feature_values = self.feature_values[idx]
        return features, feature_values, labels

class PairFMData(data.Dataset):
    def __init__(self, sampled_df, feat_idx_dict, item_num, num_ng, 
                 is_training=True, sample_method='uniform'):
        """
        :param prime sampled_df: dataframe used for sampling
        :param feat_idx_dict: feature index dictionary
        :param item_num: # of item
        :param num_ng: # of negative samples
        :param is_training: whether sampled data used for training
        :param sample_method: 'uniform' discrete uniform 
                              'item-desc' descending item popularity, high popularity means high probability to choose
                              'item-ascd' ascending item popularity, low popularity means high probability to choose
        """
        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        neg_sample_pool = list(range(item_num))
        if sample_method != 'uniform':
            popularity_item_list = sampled_df['item'].value_counts().index.tolist()
            if sample_method == 'item-desc':
                neg_sample_pool = popularity_item_list
            elif sample_method == 'item-ascd':
                neg_sample_pool = popularity_item_list[::-1]

        self.features = []
        self.feature_values = []
        self.labels = []

        if is_training:
            pair_pos = set()
            for _, row in sampled_df.iterrows():
                pair_pos.add((int(row['user']), int(row['item'])))
            print('Finish build positive matrix......')

        # construct whole data with negative sampling
        for _, row in sampled_df.iterrows():
            u, i = int(row['user']), int(row['item'])
            if is_training:
                # negative samplings
                for _ in range(num_ng):
                    if sample_method == 'uniform':
                        j = np.random.randint(item_num)
                        while (u, j) in pair_pos:
                            j = np.random.randint(item_num)
                    elif sample_method in ['item-desc', 'item-ascd']:
                        idx = 0
                        j = int(neg_sample_pool[idx])
                        while (u, j) in pair_pos:
                            idx += 1
                            j = int(neg_sample_pool[idx])
                    r = np.float32(1)  # guarantee r_{ui} >_u r_{uj}
                    # TODO if you get a more detail feature dataframe, you need to optimize this part
                    self.features.append([np.array([u + feat_idx_dict['user'], 
                                                    i + feat_idx_dict['item']], dtype=np.int64), 
                                          np.array([u + feat_idx_dict['user'], 
                                                    j + feat_idx_dict['item']], dtype=np.int64)])
                    self.feature_values.append([np.array([1, 1], dtype=np.float32), 
                                                np.array([1, 1], dtype=np.float32)])

                    self.labels.append(np.array(r))
                    
            else:
                j = i
                r = np.float32(1)  # guarantee r_{ui} >_u r_{uj}
                # TODO if you get a more detail feature dataframe, you need to optimize this part
                self.features.append([np.array([u + feat_idx_dict['user'], 
                                                i + feat_idx_dict['item']], dtype=np.int64), 
                                     np.array([u + feat_idx_dict['user'], 
                                               j + feat_idx_dict['item']], dtype=np.int64)])
                self.feature_values.append([np.array([1, 1], dtype=np.float32), 
                                            np.array([1, 1], dtype=np.float32)])
                self.labels.append(np.array(r))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features
        feature_values = self.feature_values
        labels = self.labels

        features_i = features[idx][0]
        features_j = features[idx][1]

        feature_values_i = feature_values[idx][0]
        feature_values_j = feature_values[idx][1]

        labels = labels[idx]

        return features_i, feature_values_i, features_j, feature_values_j, labels


class PairMFData(data.Dataset):
    def __init__(self, sampled_df, user_num, item_num, num_ng, is_training=True, sample_method='uniform'):
        """
        :param sampled_df: prime dataframe used for sampling
        :param user_num: # of user
        :param item_num: # of item
        :param num_ng: # of negative samples
        :param is_training: whether sampled data used for training
        :param sample_method: 'uniform' discrete uniform 
                              'item-desc' descending item popularity, high popularity means high probability to choose
                              'item-ascd' ascending item popularity, low popularity means high probability to choose
        """
        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        neg_sample_pool = list(range(item_num))
        if sample_method != 'uniform':
            popularity_item_list = sampled_df['item'].value_counts().index.tolist()
            if sample_method == 'item-desc':
                neg_sample_pool = popularity_item_list
            elif sample_method == 'item-ascd':
                neg_sample_pool = popularity_item_list[::-1]

        super(PairMFData, self).__init__()
        self.is_training = is_training
        self.num_ng = num_ng
        self.sample_num = len(sampled_df)
        self.features_fill = []

        if is_training:
            pair_pos = sp.dok_matrix((user_num, item_num), dtype=np.float32)
            for _, row in sampled_df.iterrows():
                pair_pos[int(row['user']), int(row['item'])] = 1.0
            print('Finish build positive matrix......')

        for _, row in sampled_df.iterrows():
            u, i = int(row['user']), int(row['item'])
            if is_training:
                # negative samplings
                for _ in range(num_ng):
                    if sample_method == 'uniform':
                        j = np.random.randint(item_num)
                        while (u, j) in pair_pos:
                            j = np.random.randint(item_num)
                    elif sample_method in ['item-ascd', 'item-desc']:
                        idx = 0
                        j = int(neg_sample_pool[idx])
                        while (u, j) in pair_pos:
                            idx += 1
                            j = int(neg_sample_pool[idx])
                    j = int(j)
                    r = np.float32(1)  # guarantee r_{ui} >_u r_{uj}
                    self.features_fill.append([u, i, j, r])
            else:
                r = np.float32(1)
                self.features_fill.append([u, i, i, r])

        if is_training:
            print(f'Finish negative samplings, sample number is {len(self.features_fill)}......')
    
    def __len__(self):
        return self.num_ng * self.sample_num if self.is_training else self.sample_num

    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        label = features[idx][3]

        return user, item_i, item_j, label

""" Item2Vec Specific Process """
class BuildCorpus(object):
    def __init__(self, corpus_df, window=None, max_item_num=20000, unk='<UNK>'):
        # if window is None, means no timestamp, then set max series length as window size
        bad_window = corpus_df.groupby('user')['item'].count().max()
        self.window = bad_window if window is None else window
        self.max_item_num = max_item_num
        self.unk = unk

        # build corpus
        self.corpus = corpus_df.groupby('user')['item'].apply(lambda x: x.values.tolist()).reset_index()

    def skipgram(self, record, i):
        iitem = record[i]
        left = record[max(i - self.window, 0): i]
        right = record[i + 1: i + 1 + self.window]
        return iitem, [self.unk for _ in range(self.window - len(left))] + \
                        left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self):
        max_item_num = self.max_item_num
        corpus = self.corpus
        print('building vocab...')
        self.wc = {self.unk : 1}
        for _, row in corpus.iterrows():
            sent = row['item']
            for item in sent:
                self.wc[item] = self.wc.get(item, 0) + 1

        # self.idx2item = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_item_num - 1]
        self.idx2item = sorted(self.wc, key=self.wc.get, reverse=True)[:max_item_num]
        self.item2idx = {self.idx2item[idx]: idx for idx, _ in enumerate(self.idx2item)}
        self.vocab = set([item for item in self.item2idx])
        print('build done')

    def convert(self, corpus_train_df):
        print('converting train by corpus build before...')
        data = []
        corpus = corpus_train_df.groupby('user')['item'].apply(lambda x: x.values.tolist()).reset_index()
        for _, row in corpus.iterrows():
            sent = []
            for item in row['item']:
                if item in self.vocab:
                    sent.append(item)
                else:
                    sent.append(self.unk)
            for i in range(len(sent)):
                iitem, oitems = self.skipgram(sent, i)
                data.append((self.item2idx[iitem], [self.item2idx[oitem] for oitem in oitems]))
        
        print('conversion done')

        return data

class PermutedSubsampledCorpus(data.Dataset):
    def __init__(self, dt, ws=None):
        if ws is not None:
            self.dt = []
            for iitem, oitems in dt:
                if random.random() > ws[iitem]:
                    self.dt.append((iitem, oitems))
        else:
            self.dt = dt

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        iitem, oitems = self.dt[idx]
        return iitem, np.array(oitems)

""" AE Specific Process """
class AEData(data.Dataset):
    def __init__(self, user_num, item_num, df):
        super(AEData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        self.R = np.zeros((user_num, item_num))
        self.mask_R = np.zeros((user_num, item_num))

        for _, row in df.iterrows():
            user, item, rating = int(row['user']), int(row['item']), row['rating']
            self.R[user, item] = float(rating)
            self.mask_R[user, item] = 1.

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        r = self.R[idx]
        mask_r = self.mask_R[idx]

        return mask_r, r
