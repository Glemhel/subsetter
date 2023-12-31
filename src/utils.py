import argparse
import yaml
import pandas as pd
import os
import numpy as np


from abc import ABC, abstractmethod
from typing import List, Tuple


class OptimizationAlgorithm(ABC):
    """
    Base class for optimization algorithms.
    """

    @abstractmethod
    def get_best_metrics(self):
        pass

    @abstractmethod
    def get_best_opt_value(self):
        pass

    @abstractmethod
    def step(self, i):
        pass


class SubsetterArgparser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Subsetter Argparser")

        # Define command-line arguments
        self.parser.add_argument(
            "--random_seed", type=int, default=42, help="Random seed (int, default 42)"
        )
        self.parser.add_argument(
            "--algorithm",
            type=str,
            choices=["pso", "ga"],
            default="pso",
            help="Algorithm (str, choices pso or ga, default pso)",
        )
        self.parser.add_argument(
            "--metrics",
            type=str,
            choices=["sammon", "kruskal"],
            default="sammon",
            help="Metrics (str, choices sammon or kruskal, default sammon)",
        )
        self.parser.add_argument(
            "--kind",
            type=str,
            choices=["function", "class"],
            default="function",
            help="Kind (str, function or class, default function)",
        )
        self.parser.add_argument(
            "--n_repos",
            type=int,
            default=None,
            help="Number of repositories to analyze, from n_start (sorted by size) (default None = all)",
        )
        self.parser.add_argument(
            "--i_repo_start",
            type=int,
            default=30,
            help="Index of respository to start with (small indices may not be feasible)",
        )
        self.parser.add_argument(
            "--config", type=str, default=None, help="Config file path"
        )
        self.parser.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu"],
            default="cuda",
            help="Device type (cuda or cpu)",
        )
        self.parser.add_argument(
            "--metrics_subset",
            nargs="+",
            type=int,
            default=[2, 5, 10],
            help="List of metrics subset sizes for analysis",
        )
        self.parser.add_argument(
            "--max_iter", type=int, default=30, help="Number of optimization steps"
        )
        self.parser.add_argument(
            "--num_particles", type=int, default=30, help="Number of particles (PSO)"
        )
        self.parser.add_argument(
            "--n_runs", type=int, default=1, help="Number of times to run optimization"
        )

    def load_config_from_yaml_(self, config_file: str) -> dict:
        """
        Load a configuration from a YAML file.

        :param config_file: The path to the YAML configuration file.

        :return: A dictionary containing the configuration loaded from the YAML file.
        """
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config

    def parse_args(self) -> argparse.Namespace:
        """
        Parse command-line arguments and optionally override with a configuration file.

        :return: An object containing the parsed command-line arguments.
        """
        args = self.parser.parse_args()
        if args.config:
            print(f"Parsing args from {args.config}")
            config = self.load_config_from_yaml_(args.config)
            # Use configuration values from the YAML file to override argparse defaults
            for arg_name, value in config.items():
                if value is not None:
                    setattr(args, arg_name, value)
        return args


def load_metrics_file(path) -> Tuple[pd.DataFrame, List[int]]:
    """
    Load a metrics dataset from a CSV file and preprocess it.

    :param path: The path to the CSV file containing the metrics dataset.

    :return: A pandas DataFrame containing the metrics data with proper data types,
             and an array of repository IDs ordered by size.
    """
    data = pd.read_csv(path, dtype=np.float32)
    n_nans = data.isna().values.sum()
    if n_nans > 0:
        raise ValueError(f"NaNs detected, count={n_nans}. Terminating...")
    data.rename(columns={"url": "repo_id"}, inplace=True)
    data.repo_id = data.repo_id.astype(int)
    repos_order_by_size = data.repo_id.value_counts().index.values
    return data, repos_order_by_size
