import yaml
import argparse
from pathlib import Path
import shutil
from math import isnan
import os
import csv

import numpy as np


def make_report_dict(report, kind="function"):
    res_dict = (
        {}
    )  # rest_dict[n_metric] = {loss: average loss, metric_id:percent chosen}
    metrics_names_list_path = os.path.join("..", "data", f"{kind}_metrics.txt")
    with open(metrics_names_list_path, "r") as f:
        metrics_names_list = f.readlines()
    n_metrics_list = list(report[next(iter(report))].keys())
    # fill resultant dict with initial data
    for n_metrics in n_metrics_list:
        res_dict[n_metrics] = {
            "loss": {"min": None, "avg": 0, "max": None},
            "metrics_vote": {
                metric_id: 0 for metric_id in range(len(metrics_names_list))
            },
        }
    # get data from experiments
    for repo_id in report:
        repo_data = report[repo_id]
        for n_metrics in n_metrics_list:
            # select best metric
            metric_data = repo_data[n_metrics]["selected_metrics"]
            best_run_id = np.argmin(repo_data[n_metrics]["optimums"])

            # update loss
            cur_loss = min(repo_data[n_metrics]["optimums"])

            loss_min = res_dict[n_metrics]["loss"]["min"]
            loss_avg = res_dict[n_metrics]["loss"]["avg"]
            loss_max = res_dict[n_metrics]["loss"]["max"]

            loss_min = min(loss_min, cur_loss) if loss_min is not None else cur_loss
            loss_avg += cur_loss
            loss_max = max(loss_max, cur_loss) if loss_max is not None else cur_loss

            res_dict[n_metrics]["loss"]["min"] = loss_min
            res_dict[n_metrics]["loss"]["avg"] = loss_avg
            res_dict[n_metrics]["loss"]["max"] = loss_max

            # update votecount
            for metric_id in metric_data[best_run_id][:n_metrics]:
                res_dict[n_metrics]["metrics_vote"][metric_id] += 1

    # normalize loss and metrics
    for n_metrics in n_metrics_list:
        res_dict[n_metrics]["loss"]["avg"] = res_dict[n_metrics]["loss"]["avg"] / len(
            report
        )
        for metric_id in range(len(metrics_names_list)):
            res_dict[n_metrics]["metrics_vote"][metric_id] = int(
                round(
                    res_dict[n_metrics]["metrics_vote"][metric_id] / len(report) * 100.0
                )
            )

    return res_dict


"""
Order metrics for table visualization in order of decreasing 'popularity' of being chosen
in the optimization experiments.
"""


def order_metrics(res_dict, metric_ids_list):
    metrics_popularity = {x: 0 for x in metric_ids_list}
    for n_metrics, data in res_dict.items():
        best_metrics = sorted(
            [(k, v) for k, v in data["metrics_vote"].items()], key=lambda x: -x[1]
        )[:n_metrics]
        for metric, _ in best_metrics:
            metrics_popularity[metric] += 1
    best_order = sorted(
        [(k, v) for k, v in metrics_popularity.items()], key=lambda x: -x[1]
    )
    return [x[0] for x in best_order]


"""
Save to file list of best metrics selected by optimization process
"""


def get_best_metrics_id(
    report, experiment_name, n_metrics=10, kind="function", path_prefix="."
):
    print(f"Getting best {n_metrics = } for {kind = }")
    metrics_count = {}
    for repo_id in report:
        stats = report[repo_id][n_metrics]["selected_metrics"]
        best_run_id = np.argmin(report[repo_id][n_metrics]["optimums"])
        for metric in stats[best_run_id][:n_metrics]:
            if metric not in metrics_count:
                metrics_count[metric] = 0
            metrics_count[metric] += 1

    # sort metrics by decreasing number of votes
    metrics_arr = [(k, v) for k, v in metrics_count.items()]
    metrics_arr_sorted = sorted(metrics_arr, key=lambda x: -x[1])

    # select best metrics
    best_metrics = metrics_arr_sorted[:n_metrics]
    metrics_names_list_path = os.path.join("..", "data", f"{kind}_metrics.txt")
    with open(metrics_names_list_path, "r") as f:
        metrics_names_list = f.readlines()

    savepath = os.path.join(path_prefix, f"{experiment_name}-best_metrics.dat")
    print(f"Saving best metrics to {savepath}")
    with open(savepath, "w") as f:
        for i, (metric_id, vote_count) in enumerate(best_metrics, 1):
            f.write(
                f"{i}    {metric_id}    {metrics_names_list[metric_id]}    {vote_count}\n"
            )


"""
Saves data for plotting from the report dict.
In particular, saves min, avg and max loss for each subset size to .dat file.
"""


def save_plot_data(report_dict, experiment_name, path_prefix="."):
    print("Saving report data for plots")
    path = os.path.join(path_prefix, f"{experiment_name}-stats.dat")
    with open(path, "w") as f:
        # Write header
        f.write("n_metrics    min    avg   max\n")
        # Write rows
        for n, res_dict in report_dict.items():
            n_min = res_dict["loss"]["min"]
            n_avg = res_dict["loss"]["avg"]
            n_max = res_dict["loss"]["max"]
            f.write(f"{n}    {n_min}    {n_avg}    {n_max}\n")


"""
Save information about each subset value, with loss and metrics selected.
"""


def save_table_data(
    report_dict,
    experiment_name,
    metrics_subset_list,
    order_list,
    path_prefix=".",
    kind="function",
):
    save_path_ = os.path.join(path_prefix, f"{experiment_name}-optimization_report.csv")

    metrics_names_list_path = os.path.join("..", "data", f"{kind}_metrics.txt")
    with open(metrics_names_list_path, "r") as f:
        metrics_names_list = list(map(lambda x: x.strip(), f.readlines()))

    with open(save_path_, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(["Metric"] + metric_subset_list)
        csvwriter.writerow(
            [""]
            + [
                f"{report_dict[n]['loss']['avg']:.2f}"
                for n in metrics_subset_list
            ]
        )
        for metric_id in order_list:
            csvwriter.writerow(
                [metrics_names_list[metric_id]]
                + [
                    report_dict[n]["metrics_vote"][metric_id]
                    for n in metrics_subset_list
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Results converter",
        description="Converts report.yaml into a .dat file suitable for plotting",
    )
    parser.add_argument("reportfile")
    parser.add_argument("n_metrics")
    args = parser.parse_args()

    print(f"Reading report from {args.reportfile}")
    with open(args.reportfile, "r") as f:
        report = yaml.safe_load(f)

    if args.reportfile.find("struct") != -1:
        kind = "struct"
    else:
        kind = "function"

    # saving to separate folder
    experiment_folder = "-".join(Path(args.reportfile).stem.split("-")[1:4])
    savepath = os.path.join("..", "analysis", experiment_folder)
    if not os.path.isdir(savepath):  # create if not already
        os.makedirs(savepath)
    print(f"Folder with results: {savepath}")

    # create report transformed dict
    result_dict = make_report_dict(report, kind=kind)
    # compute helper lists
    metric_subset_list = list(result_dict.keys())
    metric_ids_list = list(next(iter(result_dict.items()))[1]["metrics_vote"].keys())
    # compute best order for optimization table
    best_order = order_metrics(result_dict, metric_ids_list)

    # save .dat for plots
    save_plot_data(result_dict, experiment_folder, path_prefix=savepath)
    # save optimization data for large tables
    save_table_data(
        result_dict,
        experiment_folder,
        metric_subset_list,
        best_order,
        kind=kind,
        path_prefix=savepath,
    )
    # save best metrics
    get_best_metrics_id(
        report,
        experiment_folder,
        n_metrics=int(args.n_metrics),
        kind=kind,
        path_prefix=savepath,
    )
