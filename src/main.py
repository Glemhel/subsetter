from pso import PSOFeatureSelection
from sa import SimulatedAnnealing
from utils import OptimizationAlgorithm
from metrics import Metrics, SammonError, KruskalStress, EditDistance
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import datetime

import argparse
import yaml
from utils import SubsetterArgparser
from utils import load_metrics_file
from typing import List, Tuple

import torch
import pickle
import gc
import os

from icecream import install

install()


def get_data_i(data: pd.DataFrame, i: int, kind="function") -> torch.Tensor:
    """
    Get data for a specific repository.

    :param data: DataFrame containing metrics data.
    :param i: Repository ID.
    :param kind: Type of data to retrieve (currently only 'function' is supported).
    :return: Tensor containing data for the specified repository.
    """
    if kind != "function":
        raise NotImplementedError
    datai = data[data["repo_id"] == i]
    return torch.tensor(datai.drop(columns=["repo_id"]).values)


def run_analysis(
    data: torch.Tensor,
    pbar: tqdm,
    metric_subset_size: int = 10,
    error_function: Metrics = SammonError,
    method: OptimizationAlgorithm = PSOFeatureSelection,
    seed: int = 42,
    max_iter: int = 30,
    n_runs: int = 1,
    **opt_params,
) -> Tuple[List[int], List[float]]:
    """
    Run the feature selection analysis for a given dataset, optimization method and metrics function.

    :param data: Input data.
    :param pbar: tqdm progress bar.
    :param metric_subset_size: Size of the metric subset to search for.
    :param error_function: Metric to optimize (e.g., SammonError or KruskalStress).
    :param method: Optimization algorithm (e.g., PSOFeatureSelection or SimulatedAnnealing).
    :param seed: Random seed.
    :param max_iter: Maximum number of optimization iterations.
    :param n_runs: Number of optimization runs.
    :param opt_params: Additional optimization parameters.
    :return: Two lists, selected_metrics and optimum values.
    """
    torch.manual_seed(seed)
    selected_metrics_arr = []
    optimums_arr = []
    # run optimization algorithm n_runs times
    for i in range(n_runs):
        if i > 0:
            pbar.set_postfix_str(
                f"N_metrics: {metric_subset_size}; Run: {i+1}/{n_runs}; Opt: {optimums_arr[-1]:4f}"
            )
        else:
            pbar.set_postfix_str(
                f"N_metrics: {metric_subset_size}; Run: {i+1}/{n_runs};"
            )
        optimizer = method(
            data, error_function, n_metrics=metric_subset_size, **opt_params
        )
        # run each iteration
        for i in range(max_iter):
            optimizer.step(i)

        # get selected metrics
        selected_metrics = optimizer.get_best_metrics()
        selected_metrics_arr.append(selected_metrics.tolist())
        # get optimized value for this subset
        optimum_value = optimizer.get_best_opt_value()
        optimums_arr.append(optimum_value.tolist())

        # clear caches
        optimizer = None
        torch.cuda.empty_cache()
        gc.collect()

    return selected_metrics_arr, optimums_arr


def main(args: argparse.Namespace):
    """
    Main function for running the feature selection analysis.

    :param args: Command-line arguments.
    """
    res = {}  # saving experiment data

    # Get data for repositories
    if args.kind == "function":
        filename = "function_metrics.csv"
    elif args.kind == "struct":
        filename = "struct_metrics.csv"
    else:
        raise ValueError(f"Objects {args.kind} are not supported yet")
    data_path = os.path.join("..", "data", filename)
    data, indices_size_descending = load_metrics_file(data_path)
    device = torch.device(args.device)

    if args.algorithm == "pso":
        method = PSOFeatureSelection
        if args.max_iter > 100:
            raise ValueError("Too many iterations for PSO algorithm!")
    elif args.algorithm == "sa":
        method = SimulatedAnnealing
        if args.max_iter < 100:
            raise ValueError("Too few iterations for SA algorithm!")
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} is not supported yet")

    if args.metrics == "sammon":
        metrics = SammonError
    elif args.metrics == "kruskal":
        metrics = KruskalStress
    elif args.metrics == "editdist":
        metrics = EditDistance
    else:
        raise NotImplementedError(f"Metrics {args.metrics} is not supported yet")

    # results saving path
    timestamp = datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M")
    savepath = f"results{timestamp}.yaml"

    # loop over repositories
    r = (
        args.i_repo_start + args.n_repos
        if args.n_repos
        else len(indices_size_descending)
    )
    for repo_index in (pbar := trange(args.i_repo_start, r)):
        index = indices_size_descending[repo_index]
        repo_data = get_data_i(data, index, args.kind)
        repo_data = repo_data.to(device)

        repo_saved_info = {}
        # loop over metrics values
        for metric_subset_size in args.metrics_subset:
            selected_metrics_arr, optimums_arr = run_analysis(
                repo_data,
                pbar,
                metric_subset_size=metric_subset_size,
                error_function=metrics,
                method=method,
                **vars(args),
            )

            # clear caches
            torch.cuda.empty_cache()
            gc.collect()

            tosave = {
                "selected_metrics": selected_metrics_arr,
                "optimums": optimums_arr,
            }
            repo_saved_info[metric_subset_size] = tosave

        res[repo_index] = repo_saved_info

        # save results to yaml after each 5 iterations
        if (repo_index + 1) % 5 == 0:
            with open(savepath, "w") as yaml_file:
                yaml.dump(res, yaml_file, default_flow_style=False)

    print(f"All results saved to {savepath}")


if __name__ == "__main__":
    parser = SubsetterArgparser()
    args = parser.parse_args()
    main(args)
