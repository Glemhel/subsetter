import yaml
import argparse
from pathlib import Path
import shutil
from math import isnan
import os

import numpy as np

def get_best_metrics_id(report, n_metrics=10, kind='function', path_prefix='.'):
    print(f'Getting best {n_metrics = } for {kind = }')
    metrics_count = {}
    for repo_id in report:
        stats = report[repo_id][n_metrics]['selected_metrics']
        best_run_id = np.argmin(report[repo_id][n_metrics]['optimums'])
        for metric in stats[best_run_id][:n_metrics]:
            if metric not in metrics_count:
                metrics_count[metric] = 0
            metrics_count[metric] += 1

    metrics_arr = [(k,v) for k, v in metrics_count.items()]
    metrics_arr_sorted = sorted(metrics_arr, key = lambda x: -x[1])
    
    res = metrics_arr_sorted[:n_metrics]
    metrics_names_list_path = os.path.join('..', 'data', f'{kind}_metrics.txt')
    with open(metrics_names_list_path, 'r') as f:
        metrics_names_list = f.readlines()
    
    savepath = os.path.join(path_prefix, f"{Path(args.reportfile).stem}-best_metrics.dat")
    print(f"Saving best metrics to {savepath}")
    with open(savepath, "w") as f:
        for i, (metric_id, vote_count) in enumerate(res, 1):
            f.write(f'{i}    {metric_id}    {metrics_names_list[metric_id]}    {vote_count}\n')

def extract_metrics(report, path_prefix='.'):
    print("Extracting metrics")
    transformed = {}
    for repo_id in report.keys():
        for n in report[repo_id].keys():
            optimums = report[repo_id][n]['optimums']
            cur_val = min(optimums)
            if n in transformed.keys():
                stored_min, stored_avg, stored_max = transformed[n]
                stored_min = min(cur_val, stored_min) if not isnan(cur_val) else stored_min
                stored_max = max(cur_val, stored_max) if not isnan(cur_val) else stored_max
                stored_avg += cur_val if not isnan(cur_val) else 0
                transformed[n] = [stored_min, stored_avg, stored_max]
            else:
                transformed[n] = [cur_val] * 3

    # Normalize averages
    n_repos = len(report.keys())
    for n in transformed.keys():
        transformed[n][1] /= n_repos
   
    print("Saving transformed report")
    savepath = os.path.join(path_prefix, f"{Path(args.reportfile).stem}.dat")
    with open(savepath, "w") as f:
        # Write header
        f.write("n_metrics    min    avg   max\n")
        for n in transformed.keys():
            n_min, n_avg, n_max = transformed[n]
            f.write(f'{n}    {n_min}    {n_avg}    {n_max}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Results converter",
        description="Converts report.yaml into a .dat file suitable for plotting"
    )
    parser.add_argument('reportfile')
    args = parser.parse_args()
    
    print(f"Reading report from {args.reportfile}")
    with open(args.reportfile, 'r') as f:
        report = yaml.safe_load(f)

    if args.reportfile.find('struct') != -1:
        kind = 'struct'
    else:
        kind = 'function'

    # saving to separate folder
    subfolder = '-'.join(Path(args.reportfile).stem.split('-')[1:4])
    experiment_folder = Path(args.reportfile).stem
    savepath = os.path.join('..', 'analysis', subfolder, experiment_folder)
    if os.path.isdir(savepath):
        raise FileExistsError('Folder already exists. Aborting...')
    os.makedirs(savepath)
    print(f"Saving processed results to folder {savepath}")
    shutil.copyfile(args.reportfile, os.path.join(savepath, Path(args.reportfile).name))

    get_best_metrics_id(report, kind=kind, path_prefix=savepath)
    extract_metrics(report, path_prefix=savepath)
