import re
from prettytable import PrettyTable

from daisy.utils.sampler import Sampler
from daisy.utils.data import PointData, PairData, UAEData
from daisy.utils.loader import get_ur

def param_extract(args):
    param_set = [
        'batch_size', 'dropout', 'factors', 'lr',
        'num_layers', 'num_ng', 'reg_1', 'reg_2',
    ]

    print('Decide which parameter you want to tune')
    bar = 0
    for param in param_set:
        bar += 1
        print(f'{bar}:\t', param)
    tar_param_idx = input('type the corresponding number to choose parameter, space as delimeter:')
    tar_param_idx = tar_param_idx.strip().split(' ')

    # check if index is legal
    for idx in tar_param_idx:
        idx = int(idx)
        if len(param_set) < idx or idx < 0:
            raise ValueError('Invalid parameter index')

    param_limit = []
    for idx in tar_param_idx:
        param_limit.append(param_set[int(idx) - 1])

    return param_limit

def confirm_space(param_limit):
    param_dict = dict()
    for param in param_limit:
        print(f'Param [ {param} ] ')
        min_val, max_val = input('minimum value: '), input('maximum value: ')
        tp = input('confirm value type(int/float/choice) available: ')
        assert float(min_val) < float(max_val) and tp in ['int', 'float', 'choice'], 'Invalid value, please check'

        if tp == 'choice':
            # each choice type
            ct = input('choice type is (int/float): ')
            assert ct in ['int', 'float'], 'Invalid choice type'
            ct_op = eval(ct)
            cs = input('choice set(space as delimeter): ')
            fnl_cs = []
            for c in cs.strip().split(' '):
                if re.match(r'^(\d+)(\.\d+)?$', c) and ct_op(min_val) <= ct_op(c) <= ct_op(max_val):
                    fnl_cs.append(ct_op(c))
            param_dict[param] = [ct_op(min_val), ct_op(max_val), fnl_cs, tp]
        else:
            tp_op = eval(tp)
            step_size = input('step size(if no need, just press enter): ')
            if len(step_size) == 0:
                param_dict[param] = [tp_op(min_val), tp_op(max_val), 0, tp]
            elif len(step_size) != 0 and re.match(r'^(\d+)(\.\d+)?$', step_size):
                param_dict[param] = [tp_op(min_val), tp_op(max_val), float(step_size), tp]
            else:
                print('Invalid step value, please check!')

    print('\nOverview:\n')
    tb_view = PrettyTable()
    tb_view.field_names = ['Parameter Name', 'Type', 'min_val', 'maxs_val', 'step_size']

    for key, val in param_dict.items():
        if val[2] == 0:
            tb_view.add_row([key, val[3], val[0], val[1], 'None'])
        else:
            tb_view.add_row([key, val[3], val[0], val[1], val[2]])
    print(tb_view)

    return param_dict
