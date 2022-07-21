import json
import optuna
import numpy as np
from logging import getLogger

from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
from daisy.utils.splitter import TestSplitter, ValidationSplitter
from daisy.utils.utils import get_history_matrix, get_adj_mat, get_ur, build_candidates_set, ensure_dir
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.metrics import MAP, NDCG, Recall, Precision, HR, MRR
from daisy.model.KNNCFRecommender import ItemKNNCF
from daisy.model.PureSVDRecommender import PureSVD
from daisy.model.SLiMRecommender import SLiM
from daisy.model.PopRecommender import MostPop
from daisy.model.MFRecommender import MF
from daisy.model.FMRecommender import FM
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.model.NeuMFRecommender import NeuMF
from daisy.model.NFMRecommender import NFM
from daisy.model.NGCFRecommender import NGCF
from daisy.model.VAECFRecommender import VAECF
from daisy.model.EASERecommender import EASE

model_config = {
    'mostpop': MostPop,
    'slim': SLiM,
    'itemknn': ItemKNNCF,
    'puresvd': PureSVD,
    'mf': MF,
    'fm': FM,
    'ngcf': NGCF,
    'neumf': NeuMF,
    'nfm': NFM,
    'multi-vae': VAECF,
    'item2vec': Item2Vec,
    'ease': EASE,
}

metrics_config = {
    "recall": Recall,
    "mrr": MRR,
    "ndcg": NDCG,
    "hr": HR,
    "map": MAP,
    "precision": Precision,
}

tune_params_config = {
    'mostpop': [],
    'itemknn': ['maxk'],
    'puresvd': ['factors'],
    'slim': ['alpha', 'elastic'],
    'mf': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'fm': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'neumf': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'nfm': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
    'ngcf': ['num_ng', 'factors', 'node_dropout', 'mess_dropout', 'batch_size', 'lr', 'reg_1', 'reg_2'],
    'multi-vae': ['latent_dim', 'dropout','batch_size', 'lr', 'anneal_cap'],
    'ease': ['reg'],
    'item2vec': ['context_window', 'rho', 'lr', 'factors'],
}

param_type_config = {
    'num_layers': 'int',
    'maxk': 'int',
    'factors': 'int',
    'alpha': 'float',
    'elastic': 'float',
    'num_ng': 'int',
    'lr': 'float',
    'batch_size': 'int',
    'reg_1': 'float',
    'reg_2': 'float',
    'dropout': 'float',
    'node_dropout': 'float',
    'mess_dropout': 'float',
    'latent_dim': 'int',
    'anneal_cap': 'float',
    'reg': 'float',
    'context_window': 'int',
    'rho': 'float'
}

if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config()

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])

    ''' init logger '''
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger

    ''' unpack hyperparameters to tune '''
    param_dict = json.loads(config['hyperopt_pack'])
    algo_name = config['algo_name']
    kpi_name = config['optimization_metric']
    tune_param_names = tune_params_config[algo_name]

    ''' open logfile to record tuning process '''
    # begin tuning here
    tune_log_path = './tune_res/'
    ensure_dir(tune_log_path)

    f = open(tune_log_path + f"best_params_{config['loss_type']}_{config['algo_name']}_{config['dataset']}_{config['prepro']}_{config['val_method']}.csv", 'w', encoding='utf-8')
    line = ','.join(tune_param_names) + f',{kpi_name}'
    f.write(line + '\n')
    f.flush()

    ''' Test Process for Metrics Exporting '''
    reader, processor = RawDataReader(config), Preprocessor(config)
    df = reader.get_data()
    df = processor.process(df)
    user_num, item_num = processor.user_num, processor.item_num

    config['user_num'] = user_num
    config['item_num'] = item_num

    ''' Train Test split '''
    splitter = TestSplitter(config)
    train_index, test_index = splitter.split(df)
    train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

    ''' define optimization target function '''
    def objective(trial):
        for param in tune_param_names:
            if param_type_config[param] == 'int' and param_type_config[param]['step'] is None:
                step = param_dict[param]['step']
                config[param] = trial.suggest_int(
                    param, param_dict[param]['min'], param_dict[param]['max'], 1 if step is None else step)
            elif param_type_config[param] == 'float':
                config[param] = trial.suggest_float(
                    param, param_dict[param]['min'], param_dict[param]['max'], param_dict[param]['step'])
            else:
                raise ValueError(f'Invalid parameter type for {param}...')
        
        splitter = ValidationSplitter(config)
        cnt, kpis = 1, []
        for train_index, val_index in splitter.split(train_set):
            train, validation = train_set.iloc[train_index, :].copy(), train_set.iloc[val_index, :].copy()

            ''' get ground truth '''
            val_ur = get_ur(validation)
            train_ur = get_ur(train)
            config['train_ur'] = train_ur

            ''' build and train model '''
            if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop']:
                model = model_config[config['algo_name']](config)
                model.fit(train)
            
            elif config['algo_name'].lower() in ['multi-vae']:
                history_item_id, history_item_value, _  = get_history_matrix(train, config, row='user')
                config['history_item_id'], config['history_item_value'] = history_item_id, history_item_value
                model = model_config[config['algo_name']](config)
                train_dataset = AEDataset(train, yield_col=config['UID_NAME'])
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader)

            elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf']:
                if config['algo_name'].lower() == 'ngcf':
                    _, norm_adj, _ = get_adj_mat(user_num,item_num)
                    config['norm_adj'] = norm_adj

                model = model_config[config['algo_name']](config)
                sampler = BasicNegtiveSampler(train, config)
                train_samples = sampler.sampling()
                train_dataset = BasicDataset(train_samples)
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader)

            elif config['algo_name'].lower() in ['item2vec']:
                model = model_config[config['algo_name']](config)
                sampler = SkipGramNegativeSampler(train, config)
                train_samples = sampler.sampling()
                train_dataset = BasicDataset(train_samples)
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                model.fit(train_loader)
            else:
                raise NotImplementedError('Something went wrong when building and training...')
            logger.info(f'Finish {cnt} train-validation experiment(s)...')
            cnt += 1

            ''' build candidates set '''
            logger.info('Start Calculating Metrics...')
            val_u, val_ucands = build_candidates_set(val_ur, train_ur, config)

            ''' get predict result '''
            logger.info('==========================')
            logger.info('Generate recommend list...')
            logger.info('==========================')
            val_dataset = CandidatesDataset(val_ucands)
            val_loader = get_dataloader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
            preds = model.rank(val_loader) 

            ''' calculating KPIs '''
            kpi = metrics_config[kpi_name](val_ur, preds, val_u)
            kpis.append(kpi)
        logger.info('Finish one trial...')

        return np.mean(kpis)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2022))
    study.optimize(objective, n_trials=config['hyperopt_trail'])

    logger.info(f'Trial {study.best_trial.number} get the best {kpi_name}({study.best_trial.value}) with params: {study.best_trial.params}')
    line = ','.join([study.best_params[param] for param in tune_param_names]) + f',{study.best_value:.4f}\n'
    f.write(line)
    f.flush()
    f.close()
