from pso import PSOFeatureSelection
from metrics import SammonError
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import datetime

import argparse
import yaml
from utils import SubsetterArgparser
from utils import load_metrics_file

import torch
import pickle
import gc
import os

from icecream import ic


def get_data_i(data, i, kind='function'):
    if kind != 'function':
        raise NotImplementedError
    datai = data[data['repo_id'] == i]
    return torch.tensor(datai.drop(columns=['repo_id']).values)

def run_analysis(data,
                 metric_subset_size=10,
                 error_function=SammonError,
                 method=PSOFeatureSelection,
                 device='cpu',
                 seed=42,
                 num_particles=30,
                 max_iterations=30,
                 n_runs=1,
                 ):

    torch.manual_seed(seed)
    selected_metrics_arr = []
    optimums_arr = []
    for _ in range(n_runs):
        optimizer = method(data, num_particles, error_function, n_metrics=metric_subset_size, device=device)
        for _ in range(max_iterations):
            optimizer.step()

        selected_metrics = optimizer.get_best_metrics()
        selected_metrics_arr.append(selected_metrics.tolist())
        optimum_value = optimizer.get_best_opt_value()
        optimums_arr.append(optimum_value.tolist())

    # ic(selected_metrics_arr)
    # ic(optimums_arr)

    return selected_metrics_arr, optimums_arr


def main(args: argparse.Namespace):
    res = {} # saving experiment data

    data_path = os.path.join('..', 'data', 'function_metrics.csv')
    data, indices_size_descending = load_metrics_file(data_path)
    device = torch.device(args.device)

    if args.algorithm == 'pso':
        method = PSOFeatureSelection
    else:
        raise NotImplementedError(f'Algorithm {args.algorithm} is not supported yet')
    
    if args.metrics == 'sammon':
        metrics = SammonError
    else:
        raise NotImplementedError(f'Metrics {args.metrics} is not supported yet')
    # loop over repositories
    r = args.i_repo_start + args.n_repos if args.n_repos else len(indices_size_descending)
    # ic(args)
    # ic(r)
    for repo_index in trange(args.i_repo_start, r):
        index = indices_size_descending[repo_index]
        repo_data = get_data_i(data, index, args.kind)
        repo_data = repo_data.to(device)

        repo_saved_info = {}
        # loop over metrics values
        for metric_subset_size in args.metrics_subset:
            selected_metrics_arr, optimums_arr = run_analysis(repo_data,
                 metric_subset_size=metric_subset_size,
                 error_function=metrics,
                 method=method,
                 device=args.device,
                 seed=args.random_seed,
                 num_particles=args.num_particles,
                 max_iterations=args.max_iter,
                 n_runs=args.n_runs,
            )
            tosave = {
                'selected_metrics' : selected_metrics_arr,
                'optimums': optimums_arr,
            }
            repo_saved_info[metric_subset_size] = tosave
        
        res[repo_index] = repo_saved_info
        torch.cuda.empty_cache()
        gc.collect()
    
    # save results properly
    timestamp = datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M")
    savepath = f'results{timestamp}.yaml'
    with open(savepath, 'w') as yaml_file:
        yaml.dump(res, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    parser = SubsetterArgparser()
    args = parser.parse_args()
    main(args)
