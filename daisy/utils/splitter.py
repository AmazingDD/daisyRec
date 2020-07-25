import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit

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

