import numpy as np
from sklearn.model_selection import KFold

class TestSplitter(object):
    def __init__(self, config):
        self.test_method = config['test_method']
        self.test_size = config['test_size']
        self.uid = config['UID_NAME']
        self.tid = config['TID_NAME']

    def split(self, df):
        train_index, test_index = split_test(df, self.test_method, self.test_size, self.uid, self.tid)

        return train_index, test_index

class ValidationSplitter(object):
    def __init__(self, config):
        self.val_method = config['val_method']
        self.fold_num = config['fold_num']
        self.val_size = config['val_size']
        self.uid = config['UID_NAME']
        self.tid = config['TID_NAME']

    def split(self, df):
        train_val_index_zip = split_validation(df, self.val_method, self.fold_num, self.val_size, self.uid, self.tid)

        return train_val_index_zip

def split_test(df, test_method='rsbr', test_size=.2, uid='user', tid='timestamp'):
    """
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, way to split test set
                    'rsbr': random split by ratio
                    'tsbr': timestamp split by ratio  
                    'tloo': timestamp leave one out 
                    'rloo': random leave one out
                    'ufo': split by ratio in user level
                    'utfo': time-aware split by ratio in user level
    test_size : float, size of test set

    Returns
    -------
    train_ids : np.array index for training dataset
    test_ids : np.array index for test dataset

    """
    if test_method == 'ufo':
        test_ids = df.groupby(uid).apply(
            lambda x: x.sample(frac=test_size).index
        ).explode().values
        train_ids = np.setdiff1d(df.index.values, test_ids)

    elif test_method == 'utfo':
        # make sure df already been sorted by timestamp
        # df = df.sort_values([tid]).reset_index(drop=True)
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - test_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))

        test_ids = df.groupby(uid).apply(time_split).explode().values
        train_ids = np.setdiff1d(df.index.values, test_ids)

    elif test_method == 'tsbr':
        split_idx = int(np.ceil(len(df) * (1 - test_size)))
        train_ids, test_ids = np.arange(split_idx), np.arange(split_idx, len(df))

    elif test_method == 'rsbr':
        # train_set, test_set = train_test_split(df, test_size=test_size, random_state=2019)
        test_ids = np.random.choice(df.index.values, size=int(len(df) * test_size), replace=False)
        train_ids = np.setdiff1d(df.index.values, test_ids)

    elif test_method == 'tloo': # utloo
        df['rank_latest'] = df.groupby([uid])[tid].rank(method='first', ascending=False)
        train_ids, test_ids = df.index.values[df['rank_latest'] > 1], df.index.values[df['rank_latest'] == 1]
        del df['rank_latest']

    elif test_method == 'rloo': # urloo
        test_ids = df.groupby([uid]).apply(lambda grp: np.random.choice(grp.index))
        train_ids = np.setdiff1d(df.index.values, test_ids)

    else:
        raise ValueError('Invalid data_split value, expect: rloo, rsbr, tloo, tsbr')

    return train_ids, test_ids


def split_validation(train_set, val_method='rsbr', fold_num=1, val_size=.1, uid='user', tid='timestamp'):
    """
    method of split data into training data and validation data.

    Parameters
    ----------
    train_set : pd.DataFrame train set waiting for split validation
    val_method : str, way to split validation
                    'cv': combine with fold_num => fold_num-CV
                    'rsbr': combine with fold_num & val_size => fold_num-Split by ratio(9:1)
                    'tsbr': Split by ratio with timestamp, combine with val_size => 1-Split by ratio(9:1)
                    'tloo': Leave one out with timestamp => 1-Leave one out
                    'rloo': combine with fold_num => fold_num-Leave one out
                    'ufo': split by ratio in user level with K-fold
                    'utfo': time-aware split by ratio in user level
    fold_num : int, the number of folder need to be validated, only work when val_method is 'cv', 'rloo', or 'rsbr'
    val_size: float, the size of validation dataset

    Returns
    -------
    train_set_list : List, list of index for generated training datasets
    val_set_list : List, list of index for generated validation datasets
    cnt : cnt: int, the number of train-validation pair

    """
    train_set = train_set.reset_index(drop=True)
    
    train_set_list, val_set_list = [], []
    if val_method == 'ufo':
        for _ in range(fold_num):
            val_ids = train_set.groupby(uid).apply(
                lambda x: x.sample(frac=val_size).index
            ).explode().values
            train_ids = np.setdiff1d(train_set.index.values, val_ids)

            train_set_list.append(train_ids)
            val_set_list.append(val_ids)

    if val_method == 'utfo':
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - val_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))
        val_ids = train_set.groupby(uid).apply(time_split).explode().values
        train_ids = np.setdiff1d(train_set.index.values, val_ids)

        train_set_list.append(train_ids)
        val_set_list.append(val_ids)

    if val_method == 'cv':
        kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
        for train_ids, val_ids in kf.split(train_set):
            train_set_list.append(train_ids)
            val_set_list.append(val_ids)

    if val_method == 'rsbr':
        for _ in range(fold_num):
            val_ids = np.random.choice(train_set.index.values, size=int(len(train_set) * val_size), replace=False)
            train_ids = np.setdiff1d(train_set.index.values, val_ids)

            train_set_list.append(train_ids)
            val_set_list.append(val_ids)

    elif val_method == 'tsbr':
        split_idx = int(np.ceil(len(train_set) * (1 - val_size)))
        train_ids, val_ids = np.arange(split_idx), np.arange(split_idx, len(train_set))

        train_set_list.append(train_ids)
        val_set_list.append(val_ids)

    elif val_method == 'rloo':
        for _ in range(fold_num):
            val_ids = train_set.groupby([uid]).apply(lambda grp: np.random.choice(grp.index))

            train_ids = np.setdiff1d(train_set.index.values, val_ids)

            train_set_list.append(train_ids)
            val_set_list.append(val_ids)

    elif val_method == 'tloo':
        train_set['rank_latest'] = train_set.groupby([uid])[tid].rank(method='first', ascending=False)
        train_ids = train_set.index.values[train_set['rank_latest'] > 1]
        val_ids = train_set.index.values[train_set['rank_latest'] == 1]
        del train_set['rank_latest']

        train_set_list.append(train_ids)
        val_set_list.append(val_ids)

    return zip(train_set_list, val_set_list)
