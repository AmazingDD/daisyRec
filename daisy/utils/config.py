import os
import re
import yaml
import random
import logging
import colorlog
import numpy as np
from colorama import init

import torch
import torch.nn as nn

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

from daisy.utils.metrics import Precision, Recall, NDCG, MRR, MAP, HR, F1, AUC, Coverage, Diversity, Popularity
from daisy.utils.parser import parse_args
from daisy.utils.utils import ensure_dir, get_local_time

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
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

metrics_config = {
    "recall": Recall,
    "mrr": MRR,
    "ndcg": NDCG,
    "hr": HR,
    "map": MAP,
    "precision": Precision,
    "f1": F1,
    "auc": AUC,
    "coverage": Coverage,
    "diversity": Diversity,
    "popularity": Popularity,
}

metrics_name_config = {
    "recall": 'Recall',
    "mrr": 'MRR',
    "ndcg": 'NDCG',
    "hr": 'Hit Ratio',
    "precision": 'Precision',
    "f1": 'F1-score',
    "auc": 'AUC',
    "coverage": 'Coverage',
    "diversity": 'Diversity',
    "popularity": 'Average Popularity',
}

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

initializer_param_config = {
    'normal': {'mean':0.0, 'std':0.01},
    'uniform': {'a':0.0, 'b':1.0},
    'xavier_normal': {'gain':1.0},
    'xavier_uniform': {'gain':1.0}
}

initializer_config = {
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_
}

def init_seed(seed, reproducibility):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn

    Parameters
    ----------
    seed : int
        random seed
    reproducibility : bool
        Whether to require reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def init_config(param_dict=None):
        ''' 
        summarize hyper-parameter part (basic yaml + args + model yaml) 
        '''
        config = dict()
        basic_conf = yaml.load(open('./daisy/config/basic.yaml'), Loader=yaml.loader.SafeLoader)
        config.update(basic_conf)

        args = parse_args()
        algo_name = config['algo_name'] if args.algo_name is None else args.algo_name

        model_conf = yaml.load(
            open(f'./daisy/config/model/{algo_name}.yaml'), Loader=yaml.loader.SafeLoader)
        config.update(model_conf)

        args_conf = vars(args)
        config.update(args_conf)

        if param_dict is not None:
            config.update(param_dict)

        return config

class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True

def init_logger(config):
    init(autoreset=True)
    log_root = './log/'
    dir_name = os.path.dirname(log_root)
    ensure_dir(dir_name)
    model_name = os.path.join(dir_name, config['algo_name'])
    ensure_dir(model_name)

    logfilename = f'{config["algo_name"]}/{get_local_time()}.log'
    logfilepath = os.path.join(log_root, logfilename)

    filefmt = "%(asctime)-10s %(levelname)s - %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-10s %(levelname)s - %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])
