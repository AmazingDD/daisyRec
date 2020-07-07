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


def load_rate(src='ml-100k', prepro='origin', binary=True, pos_threshold=None, prepro_level='ui'):
    """
    method of loading certain raw data
    Parameters
    ----------
    src : str, the name of dataset
    prepro : str, way to pre-process raw data input, expect 'origin' or f'{N}core', N is integer value
    binary : boolean, whether to transform rating to binary label as CTR or not as Regression
    pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
    prepro_level : str, which level to do with 'Ncore' operation (it only works when prepro contains 'core')

    Returns
    -------
    df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
    user_num : int, the number of users
    item_num : int, the number of items
    """
    df = pd.DataFrame()
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
        df.rename(columns={'userID': 'user', 'artistID': 'item', 'weight': 'rating'}, inplace=True)
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
        # TODO this dataset has wrong source URL, we will figure out in future
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

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if prepro_level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif prepro_level == 'u':
            df = filter_user(df)
        elif prepro_level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid prepro_level value: {prepro_level}')

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


def split_test(df, test_method='fo', test_size=.2):
    """
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, way to split test set
                    'fo': split by ratio
                    'tfo': split by ratio with timestamp
                    'tloo': leave one out with timestamp
                    'loo': leave one out
                    'ufo': split by ratio in user level
                    'utfo': time-aware split by ratio in user level
    test_size : float, size of test set

    Returns
    -------
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """

    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    if test_method == 'ufo':
        driver_ids = df['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=2020)
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    elif test_method == 'utfo':
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - test_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))

        test_index = df.groupby('user').apply(time_split).explode().values
        test_set = df.loc[test_index, :]
        train_set = df[~df.index.isin(test_index)]

    elif test_method == 'tfo':
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
    method of split data into training data and validation data.
    (Currently, this method returns list of train & validation set, but I'll change 
    it to index list or generator in future so as to save memory space) TODO

    Parameters
    ----------
    train_set : pd.DataFrame train set waiting for split validation
    val_method : str, way to split validation
                    'cv': combine with fold_num => fold_num-CV
                    'fo': combine with fold_num & val_size => fold_num-Split by ratio(9:1)
                    'tfo': Split by ratio with timestamp, combine with val_size => 1-Split by ratio(9:1)
                    'tloo': Leave one out with timestamp => 1-Leave one out
                    'loo': combine with fold_num => fold_num-Leave one out
                    'ufo': split by ratio in user level with K-fold
                    'utfo': time-aware split by ratio in user level
    fold_num : int, the number of folder need to be validated, only work when val_method is 'cv', 'loo', or 'fo'
    val_size: float, the size of validation dataset

    Returns
    -------
    train_set_list : List, list of generated training datasets
    val_set_list : List, list of generated validation datasets
    cnt : cnt: int, the number of train-validation pair

    """
    if val_method in ['tloo', 'tfo', 'utfo']:
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
    if val_method == 'utfo':
        train_set = train_set.sort_values(['user', 'timestamp']).reset_index(drop=True)
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - val_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))
        val_index = train_set.groupby('user').apply(time_split).explode().values
        val_set = train_set.loc[val_index, :]
        train_set = train_set[~train_set.index.isin(val_index)]
        train_set_list.append(train_set)
        val_set_list.append(val_set)
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
    """

    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur


def get_ir(df):
    """

    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir


def build_feat_idx_dict(df:pd.DataFrame, 
                        cat_cols:list=['user', 'item'], 
                        num_cols:list=[]):
    """

    Parameters
    ----------
    df : pd.DataFrame feature dataframe
    cat_cols : List, list of categorical column names
    num_cols : List, list of numeric column names

    Returns
    -------
    feat_idx_dict : Dictionary, dict with index-feature column mapping information
    cnt : int, the number of features
    """
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
    for _ in num_cols:
        cnt += 1
    print(f'Number of features: {cnt}')

    return feat_idx_dict, cnt


def convert_npy_mat(user_num, item_num, df):
    """
    method of convert dataframe to numoy matrix
    Parameters
    ----------
    user_num : int, the number of users
    item_num : int, the number of items
    df :  pd.DataFrame, rating dataframe

    Returns
    -------
    mat : np.matrix, rating matrix
    """
    mat = np.zeros((user_num, item_num))
    for _, row in df.iterrows():
        u, i, r = row['user'], row['item'], row['rating']
        mat[int(u), int(i)] = float(r)
    return mat


def build_candidates_set(test_ur, train_ur, item_pool, candidates_num=1000):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_pool : the set of all items
    candidates_num : int, the number of candidates
    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        sub_item_pool = item_pool - v - train_ur[k] # remove GT & interacted
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))
    
    return test_ucands


class Sampler(object):
    def __init__(self, user_num, item_num, num_ng=4, sample_method='item-desc', sample_ratio=0):
        """

        Parameters
        ----------
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, # of nagative sampling per sample
        sample_method : str, sampling method
                        'uniform' discrete uniform
                        'item-desc' descending item popularity, high popularity means high probability to choose
                        'item-ascd' ascending item popularity, low popularity means high probability to choose
        sample_ratio : float, scope [0, 1], it determines what extent the sample method except 'uniform' occupied
        """
        self.user_num = user_num
        self.item_num = item_num
        self.num_ng = num_ng
        self.sample_method = sample_method
        self.sample_ratio = sample_ratio

        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        assert 0 <= sample_ratio <= 1, 'Invalid sample ratio value'

    def transform(self, sampled_df, is_training=True):
        """

        Parameters
        ----------
        sampled_df : pd.DataFrame, dataframe waiting for sampling
        is_training : boolean, whether the procedure using this method is training part

        Returns
        -------
        neg_set : List, list of (user, item, rating, negative sampled items)
        """
        if not is_training:
            neg_set = []
            for _, row in sampled_df.iterrows():
                u = int(row['user'])
                i = int(row['item'])
                r = row['rating']
                js = []
                neg_set.append([u, i, r, js])
            
            return neg_set

        user_num = self.user_num
        item_num = self.item_num
        pair_pos = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for _, row in sampled_df.iterrows():
            pair_pos[int(row['user']), int(row['item'])] = 1.0

        neg_sample_pool = list(range(item_num))
        popularity_item_list = sampled_df['item'].value_counts().index.tolist()
        if self.sample_method == 'item-desc':
            neg_sample_pool = popularity_item_list
        elif self.sample_method == 'item-ascd':
            neg_sample_pool = popularity_item_list[::-1]
        
        neg_set = []
        uni_num = int(self.num_ng * (1 - self.sample_ratio))
        ex_num = self.num_ng - uni_num
        for _, row in sampled_df.iterrows():
            u = int(row['user'])
            i = int(row['item'])
            r = row['rating']

            js = []
            for _ in range(uni_num):
                j = np.random.randint(item_num)
                while (u, j) in pair_pos:
                    j = np.random.randint(item_num)
                js.append(j)
            for _ in range(ex_num):
                if self.sample_method in ['item-desc', 'item-ascd']:
                    idx = 0
                    j = int(neg_sample_pool[idx])
                    while (u, j) in pair_pos:
                        idx += 1
                        j = int(neg_sample_pool[idx])
                    js.append(j)
                else:
                    # maybe add other sample methods in future, uniform as default
                    j = np.random.randint(item_num)
                    while (u, j) in pair_pos:
                        j = np.random.randint(item_num)
                    js.append(j)
            neg_set.append([u, i, r, js])

        print(f'Finish negative samplings, sample number is {len(neg_set) * self.num_ng}......')

        return neg_set


class PointData(data.Dataset):
    def __init__(self, neg_set, is_training=True, neg_label_val=0.):
        """
        Dataset formatter adapted point-wise algorithms
        Parameters
        ----------
        neg_set : List, negative sampled result generated by Sampler
        is_training : boolean, whether the procedure using this method is training part
        neg_label_val : float, rating value towards negative sample
        """
        super(PointData, self).__init__()
        self.features_fill = []
        self.labels_fill = []
        for u, i, r, js in neg_set:
            self.features_fill.append([int(u), int(i)])
            self.labels_fill.append(r)
            
            if is_training:
                for j in js:
                    self.features_fill.append([int(u), int(j)])
                    self.labels_fill.append(neg_label_val)
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


class PairData(data.Dataset):
    def __init__(self, neg_set, is_training=True):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        neg_set : List,
        is_training : bool,
        """
        super(PairData, self).__init__()
        self.features_fill = []

        for u, i, r, js in neg_set:
            u, i, r = int(u), int(i), np.float32(1)
            if is_training:
                for j in js:
                    self.features_fill.append([u, i, j, r])
            else:
                self.features_fill.append([u, i, i, r])

    def __len__(self):
        return len(self.features_fill)

    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        label = features[idx][3]

        return user, item_i, item_j, label


class UAEData(data.Dataset):
    def __init__(self, user_num, item_num, train_set, test_set):
        """
        user-level Dataset formatter adapted AutoEncoder-like algorithms
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        train_set : pd.DataFrame, training set
        test_set : pd.DataFrame, test set
        """
        super(UAEData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        self.R = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # true label
        self.mask_R = sp.dok_matrix((user_num, item_num), dtype=np.float32) # only concern interaction known
        self.user_idx = np.array(range(user_num))

        for _, row in train_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[user, item] = 1.
            self.mask_R[user, item] = 1.

        for _, row in test_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[user, item] = 1.

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        u = self.user_idx[idx]
        ur = self.R[idx].A.squeeze()
        mask_ur = self.mask_R[idx].A.squeeze()

        return u, ur, mask_ur


class IAEData(data.Dataset):
    def __init__(self, user_num, item_num, train_set, test_set):
        """
        item-level Dataset formatter adapted AutoEncoder-like algorithms
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        train_set : pd.DataFrame, training set
        test_set : pd.DataFrame, test set
        """
        super(IAEData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        
        self.R = sp.dok_matrix((item_num, user_num), dtype=np.float32)  # true label
        self.mask_R = sp.dok_matrix((item_num, user_num), dtype=np.float32) # only concern interaction known
        self.item_idx = np.array(range(item_num))

        for _, row in train_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[item, user] = 1.
            self.mask_R[item, user] = 1.

        for _, row in test_set.iterrows():
            user, item = int(row['user']), int(row['item'])
            self.R[item, user] = 1.

    def __len__(self):
        return self.item_num

    def __getitem__(self, idx):
        i = self.item_idx[idx]
        ir = self.R[idx].A.squeeze()
        mask_ir = self.mask_R[idx].A.squeeze()

        return i, ir, mask_ir


class BuildCorpus(object):
    def __init__(self, corpus_df, window=None, max_item_num=20000, unk='<UNK>'):
        """
        Item2Vec Specific Process, building item-corpus by dataframe
        Parameters
        ----------
        corpus_df : pd.DataFrame, the whole dataset
        window : int, window size
        max_item_num : the maximum item pool size,
        unk : str, if there are items beyond existed items, they will all be treated as this value
        """
        # if window is None, means no timestamp, then set max series length as window size
        bad_window = corpus_df.groupby('user')['item'].count().max()
        self.window = bad_window if window is None else window
        self.max_item_num = max_item_num
        self.unk = unk

        # build corpus
        self.corpus = corpus_df.groupby('user')['item'].apply(lambda x: x.values.tolist()).reset_index()

        self.wc = None
        self.idx2item = None
        self.item2idx = None
        self.vocab = None

    def skip_gram(self, record, i):
        iitem = record[i]
        left = record[max(i - self.window, 0): i]
        right = record[i + 1: i + 1 + self.window]
        return iitem, [self.unk for _ in range(self.window - len(left))] + \
                        left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self):
        max_item_num = self.max_item_num
        corpus = self.corpus
        print('building vocab...')
        self.wc = {self.unk: 1}
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
        """

        Parameters
        ----------
        corpus_train_df

        Returns
        -------
        dt
        """
        print('converting train by corpus build before...')
        dt = []
        corpus = corpus_train_df.groupby('user')['item'].apply(lambda x: x.values.tolist()).reset_index()
        for _, row in corpus.iterrows():
            sent = []
            for item in row['item']:
                if item in self.vocab:
                    sent.append(item)
                else:
                    sent.append(self.unk)
            for i in range(len(sent)):
                iitem, oitems = self.skip_gram(sent, i)
                dt.append((self.item2idx[iitem], [self.item2idx[oitem] for oitem in oitems]))
        
        print('conversion done')

        return dt


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


def get_weights(wc, idx2item, ss_t, whether_weights):
    wf = np.array([wc[item] for item in idx2item])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2item)
    weights = wf if whether_weights else None

    return vocab_size, weights


def item2vec_data(train_set, test_set, window, item_num, batch_size, ss_t=1e-5, unk='<UNK>', weights=None):
    """

    Parameters
    ----------
    train_set : pd.DataFrame,
    test_set : pd.DataFrame,
    window : int, rolling window size
    item_num : int, the number of total items
    batch_size : batch size
    ss_t : float
    unk : str,
    weights : wheter parse weight

    Returns
    -------
    data_loader: torch.data.Dataset, data generator used for Item2Vec
    vocab_size: int, max item length
    pre.item2idx, dict, the mapping information for item to index code
    """
    df = pd.concat([train_set, test_set], ignore_index=True)
    pre = BuildCorpus(df, window, item_num + 1, unk)
    pre.build()

    dt = pre.convert(train_set)
    vocab_size, weights = get_weights(pre.wc, pre.idx2item, ss_t, weights)
    data_set = PermutedSubsampledCorpus(dt)  
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True) 

    return data_loader, vocab_size, pre.item2idx
