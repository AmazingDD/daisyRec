import os
import gc
from daisy.utils.loader import load_rate, split_test

# 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'bx', 'amazon-cloth',
# 'amazon-electronic', 'amazon-book', 'amazon-music', 'epinions', 'yelp', 'citeulike'

dataset_list = ['netflix']

prepro_list = ['5core', '10core'] # 'origin', 

if not os.path.exists('./experiment_data/'):
    os.makedirs('./experiment_data/')

for dataset in dataset_list:
    print(dataset)
    for prepro in prepro_list:
        print(prepro)
        df, user_num, item_num = load_rate(dataset, prepro, False)
        for test_method in ['fo', 'loo', 'tloo', 'tfo']:
            print(test_method)
            train_set, test_set = split_test(df, test_method, .2)
            train_set.to_csv(f'./experiment_data/train_{dataset}_{prepro}_{test_method}.dat', index=False)
            test_set.to_csv(f'./experiment_data/test_{dataset}_{prepro}_{test_method}.dat', index=False)

            
            print('Finish save train and test set......')

        del train_set, test_set, df
        gc.collect()