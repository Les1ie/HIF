import json
import pickle
import warnings
from copy import deepcopy
from functools import partial
from os import path as osp
from datetime import datetime
import os
import json
# from ax.plot.contour import interact_contour
# from ax.utils.notebook.plotting import render
import ax.plot.contour
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.utils.testing.core_stubs import get_branin_search_space

from run import parse_args, run
from ax import optimize

global_args = parse_args()


def tune(p, ):
    local_args = deepcopy(global_args)
    for k, v in p.items():
        setattr(local_args, k, v)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run(local_args)
    if isinstance(result, list):
        result = result[0]
    msle = result['hp_metric']
    return {'msle': (msle, 0.0)}


def save_tune_result(results):
    now = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    save_dir = 'tune_results'
    save_name = f'{now}.pkl'
    js = {'best_parameters': results[0],
          'best_result': results[1][0]}
    json_name = f'{now}.json'
    save_path = osp.join(save_dir, save_name)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    with open(osp.join(save_dir, json_name), 'w', encoding='utf8') as f:
        json.dump(js, f, indent=2)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parameters = [
        # 神经网络参数
        # {'name': 'learning_rate', 'type': 'range', 'value_type': 'float', 'bounds': [5e-5, 5e-1], 'log_scale': True},
        # {'name': 'weight_decay', 'type': 'range', 'value_type': 'float', 'bounds': [5e-5, 5e-1], 'log_scale': True},
        # {'name': 'noise_dim', 'type': 'choice', 'value_type': 'int', 'values': [0, 1]},
        # {'name': 'noise_rate', 'type': 'range', 'value_type': 'float', 'bounds': [0, 1]},
        # {'name': 'noise_weight', 'type': 'range', 'value_type': 'float', 'bounds': [0, 1]},
        # {'name': 'bro_loss_weight', 'type': 'range', 'value_type': 'float', 'bounds': [0, 1], 'log_scale': True},
        # {'name': 'time_loss_weight', 'type': 'choice', 'value_type': 'float', 'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 0],},
        # {'name': 'l1_weight', 'type': 'choice', 'value_type': 'float', 'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 0],},
        # {'name': 'in_feats', 'type': 'choice', 'value_type': 'int', 'values': [16, 32, 64], 'is_ordered': True, },
        # {'name': 'num_heads', 'type': 'choice', 'value_type': 'int', 'values': [8, 16], 'is_ordered': True, },
        {'name': 'num_gcn_layers', 'type': 'range', 'value_type': 'int', 'bounds': [1, 5], },
        {'name': 'num_readout_layers', 'type': 'range', 'value_type': 'int', 'bounds': [2, 4], },
        {'name': 'num_time_module_layers', 'type': 'range', 'value_type': 'int', 'bounds': [2, 6], },
        # {'name': 'share_time_module', 'type': 'choice', 'value_type': 'bool', 'values': [False, True], },
        {'name': 'time_module', 'type': 'choice', 'value_type': 'str', 'values': ['None', 'rnn', 'transformer'], },
        {'name': 'time_decay_pos', 'type': 'choice', 'value_type': 'str', 'values': ['None', 'head', 'all', 'tail'], },
        {'name': 'learnable_embedding', 'type': 'choice', 'value_type': 'bool', 'values': [True, False], },

        # {'name': 'patience', 'type': 'choice', 'value_type': 'int', 'values': [10, 20, 50], 'is_ordered': True, },
        # {'name': 'dropout', 'type': 'range', 'value_type': 'float', 'bounds': [0, 0.5], },
        # {'name': 'dropout_edge', 'type': 'range', 'value_type': 'float', 'bounds': [0, 0.5], },

        # 数据处理参数，会导致重新处理数据
        # {'name': 'num_time_nodes', 'type': 'choice', 'value_type': 'int', 'values': [5, 6, 7, 8, 9, 10], 'is_ordered': True},
        # {'name': 'soft_partition', 'type': 'range', 'value_type': 'int', 'bounds': [0, 3]},
        # {'name': 'alpha', 'type': 'choice', 'value_type': 'int', 'values': [10, 20, 40, 80, 100], 'is_ordered': True},
        # {'name': 'beta', 'type': 'choice', 'value_type': 'int', 'values': [50, 100, 150, 200], 'is_ordered': True},
        # {'name': 'sample_batch', 'type': 'range', 'value_type': 'int', 'bounds': [5, 30],},
        # {'name': 'source_base', 'type': 'range', 'value_type': 'int', 'bounds': [3, 12], },
        # {'name': 'reposted_base', 'type': 'range', 'value_type': 'int', 'bounds': [3, 12], },
        # {'name': 'leaf_base', 'type': 'range', 'value_type': 'int', 'bounds': [3, 12], },
    ]
    tune_under_global_args = partial(tune, global_args=global_args)
    best_parameters, best_values, experiment, model = optimize(
        parameters=parameters,
        # Booth function
        evaluation_function=tune,
        minimize=True,
        random_seed=0,
        total_trials=50,
        experiment_name=f'tune on {global_args.name}',
        objective_name='msle',
        # generation_strategy=choose_generation_strategy(get_branin_search_space()),
    )
    print('best parameters:', best_parameters)
    print('best result:', best_values)
    print('experiment name:', experiment)
    print('model:', model)
    # ax.plot.contour.plot_contour(model, 'learning_rate', 'weight_decay', 'hp_metric')
    save_tune_result((best_parameters, best_values, experiment))
