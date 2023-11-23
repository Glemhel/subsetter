import os
import yaml
import argparse
import random
import numpy as np
from scipy.stats import mannwhitneyu
from metrics import Metrics, SammonError, KruskalStress, EditDistance
from utils import load_metrics_file
from main import get_data_i


TRAIN_VAL_SPLIT_SEED = 42

def read_report_file(path):
    print(f"Reading report from {path}")
    with open(path, 'r') as f:
        report = yaml.safe_load(f)

    return report

def get_metrics_subset_sizes(metrics):
    metrics_subset = list(metrics[list(metrics.keys())[0]].keys()) 
    return metrics_subset

def get_voting_validation_sets(repositories, val=0.3):
    random.seed(TRAIN_VAL_SPLIT_SEED)
    val_size = int(0.3 * len(repositories))
    validation = set(random.sample(list(repositories), val_size))
    voting = repositories - validation
    return voting, validation

def get_best_metrics_id(report, repositories, n_metrics=10):
    metrics_count = {}
    for repo_id in repositories:
        stats = report[repo_id][n_metrics]['selected_metrics']
        best_run_id = np.argmin(report[repo_id][n_metrics]['optimums'])
        for metric in stats[best_run_id][:n_metrics]:
            if metric not in metrics_count:
                metrics_count[metric] = 0
            metrics_count[metric] += 1

    metrics_arr = [(k,v) for k, v in metrics_count.items()]
    metrics_arr_sorted = sorted(metrics_arr, key = lambda x: -x[1])
    res = [id for id, _ in metrics_arr_sorted[:n_metrics]]
    return res

def compute_loss(metrics, fitness_function, validation_set, metrics_subset, indices_size_descending):
    losses = []
    for repo in validation_set:
        data = get_data_i(metrics, indices_size_descending[repo])
        loss = fitness_function(data).compute(metrics_subset)
        losses.append(loss)
    return losses
    

def compute_losses(metrics, report, fitness_function, voting_set, validation_set, metrics_subset_sizes, indices_size_descending):
    losses = []
    for subset_size in metrics_subset_sizes:
        # Iterate through voting set & find the best set for a given number of metrics
        best_metrics = get_best_metrics_id(report, voting_set, n_metrics=subset_size)
        
        # Iterate through validation set with computed metrics & compute the mean value of the loss
        loss = np.mean(compute_loss(metrics, fitness_function, validation_set, best_metrics, indices_size_descending))
        losses.append(loss)
    return losses

def run_pairwise_analysis(losses_1, losses_2, metrics_subset_sizes):
    print('Subset size | Loss 1               | Loss 2')
    for subset_size, loss_1, loss_2 in zip(metrics_subset_sizes, losses_1, losses_2):
        print(f'{subset_size: 11} | {loss_1: <20} | {loss_2}')

    u1, p = mannwhitneyu(losses_1, losses_2)
    n1, n2 = len(losses_1), len(losses_2)
    u2 = n1 * n2 - u1
    print(f'U1 = {u1}, U2 = {u2}, p = {p}')

def run_detailed(metrics, report_1, report_2, fitness_function, voting_set, validation_set, metrics_subset_sizes, indices_size_descending):
    print('Subset size | U1      | U2      | p')
    for subset_size in metrics_subset_sizes:
        best_metrics_1 = get_best_metrics_id(report_1, voting_set, n_metrics=subset_size)
        best_metrics_2 = get_best_metrics_id(report_2, voting_set, n_metrics=subset_size)

        loss_1 = compute_loss(metrics, fitness_function, validation_set, best_metrics_1, indices_size_descending)
        loss_2 = compute_loss(metrics, fitness_function, validation_set, best_metrics_2, indices_size_descending)

        u1, p = mannwhitneyu(losses_1, losses_2)
        n1, n2 = len(loss_1), len(loss_2)
        u2 = n1 * n2 - u1
        print(f'{subset_size:11} | {u1: <7} | {u2: <7} | {p}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Methods comparator')
    parser.add_argument('first_method')
    parser.add_argument('second_method')
    parser.add_argument(
        '--metrics',
        type=str,
        choices=['sammon', 'kruskal', 'editdist'],
        default='sammon',
        help='Metrics (str, choices sammon, kruskal or editdist, default sammon)',
    )
    parser.add_argument(
        '--kind',
        type=str,
        choices=['function', 'class'],
        default='function',
        help='Kind (str, function or class, default function)',
    )

    args = parser.parse_args()
    fitness_function = None

    if args.metrics == "sammon":
        fitness_function = SammonError
    elif args.metrics == "kruskal":
        fitness_function = KruskalStress
    elif args.metrics == "editdist":
        fitness_function = EditDistance
    else:
        raise NotImplementedError(f"Metrics {args.metrics} is not supported yet")

    if args.kind == "function":
        filename = "function_metrics.csv"
    elif args.kind == "struct":
        filename = "struct_metrics.csv"
    else:
        raise ValueError(f"Objects {args.kind} are not supported yet")
    data_path = os.path.join("..", "data", filename)
    metrics, indices_size_descending = load_metrics_file(data_path)

    report_1 = read_report_file(args.first_method)
    report_2 = read_report_file(args.second_method)

    repos = set(report_1.keys())
    assert repos == set(report_2.keys()), 'Metrics must be computed on the same set of repositories'

    voting, validation = get_voting_validation_sets(repos)
    metrics_sets_sizes = get_metrics_subset_sizes(report_1)

    assert set(metrics_sets_sizes) == set(get_metrics_subset_sizes(report_2)), 'Metrics subsets must be the same'

    print('Computing losses based on the 1st report')
    losses_1 = compute_losses(metrics, report_1, fitness_function, voting, validation, metrics_sets_sizes, indices_size_descending)
    print('Computing losses based on the 2nd report')
    losses_2 = compute_losses(metrics, report_2, fitness_function, voting, validation, metrics_sets_sizes, indices_size_descending)
    
    run_pairwise_analysis(losses_1, losses_2, metrics_sets_sizes)
    run_detailed(metrics, report_1, report_2, fitness_function, voting, validation, metrics_sets_sizes, indices_size_descending)