import os
import re
import yaml
import torch
import random
import logging
import colorlog
import numpy as np
from colorama import init

from daisy.utils.parser import parse_args
from daisy.utils.utils import ensure_dir, get_local_time

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
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

        current_path = os.path.dirname(os.path.realpath(__file__))
        basic_init_file = os.path.join(current_path, '../assets/basic.yaml')
        
        basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
        config.update(basic_conf)

        args = parse_args()
        algo_name = config['algo_name'] if args.algo_name is None else args.algo_name
        model_init_file = os.path.join(current_path, f'../assets/{algo_name}.yaml')

        model_conf = yaml.load(
            open(model_init_file), Loader=yaml.loader.SafeLoader)
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
