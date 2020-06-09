import os
import gc
from daisy.utils.loader import load_rate, split_test


def generate_experiment_data(dataset, prepro, test_method):
    """
    method of generating dataset for reproducing paper KPI
    Parameters
    ----------
    dataset : str, dataset name, available options: 'netflix', 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'bx',
                                                    'amazon-cloth', 'amazon-electronic', 'amazon-book', 'amazon-music',
                                                    'epinions', 'yelp', 'citeulike'
    prepro : str, way to pre-process data, available options: 'origin', '5core', '10core'
    test_method : str, way to get test dataset, available options: 'fo', 'loo', 'tloo', 'tfo'

    Returns
    -------

    """
    if not os.path.exists('./experiment_data/'):
        os.makedirs('./experiment_data/')
    print(f'start process {dataset} with {prepro} method...')
    df, user_num, item_num = load_rate(dataset, prepro, False)
    print(f'test method : {test_method}')
    train_set, test_set = split_test(df, test_method, .2)
    train_set.to_csv(f'./experiment_data/train_{dataset}_{prepro}_{test_method}.dat', index=False)
    test_set.to_csv(f'./experiment_data/test_{dataset}_{prepro}_{test_method}.dat', index=False)
    print('Finish save train and test set...')
    del train_set, test_set, df
    gc.collect()
