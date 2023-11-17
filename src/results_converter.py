import yaml
import argparse
from pathlib import Path
from math import isnan
import os

def get_best_metrics_id(report, n_metrics=10, kind='function'):
    print(f'Getting best {n_metrics = } for {kind = }')
    metrics_count = {}
    for repo_id in report:
        stats = report[repo_id][n_metrics]['selected_metrics']
        for selected_metrics in stats:
            for metric in selected_metrics[:n_metrics]:
                if metric not in metrics_count:
                    metrics_count[metric] = 0
                metrics_count[metric] += 1

    metrics_arr = [(k,v) for k, v in metrics_count.items()]
    metrics_arr_sorted = sorted(metrics_arr, key = lambda x: -x[1])
    
    res = metrics_arr_sorted[:n_metrics]
    metrics_names_list_path = os.path.join('..', 'data', f'{kind}_metrics.txt')
    with open(metrics_names_list_path, 'r') as f:
        metrics_names_list = f.readlines()
    
    savepath = f"{Path(args.reportfile).stem}-best_metrics.dat"
    print(f"Saving best metrics to {savepath}")
    with open(savepath, "w") as f:
        for i, (metric_id, vote_count) in enumerate(res, 1):
            f.write(f'{i}    {metric_id}    {metrics_names_list[metric_id]}    {vote_count}\n')

def extract_metrics(report):
    print("Extracting metrics")
    transformed = {}
    for repo_id in report.keys():
        for n in report[repo_id].keys():
            optimums = report[repo_id][n]['optimums']
            cur_val = min(optimums)
            if n in transformed.keys():
                stored_min, stored_max, stored_avg = transformed[n]
                stored_min = min(cur_val, stored_min) if not isnan(cur_val) else stored_min
                stored_max = max(cur_val, stored_max) if not isnan(cur_val) else stored_max
                stored_avg += cur_val if not isnan(cur_val) else 0
                transformed[n] = [stored_min, stored_max, stored_avg]
            else:
                transformed[n] = [cur_val] * 3

    # Normalize averages
    n_repos = len(report.keys())
    for n in transformed.keys():
        transformed[n][2] /= n_repos
   
    print("Saving transformed report")
    with open(f"{Path(args.reportfile).stem}.dat", "w") as f:
        # Write header
        f.write("n_metrics    max    min   avg\n")
        for n in transformed.keys():
            n_min, n_max, n_avg = transformed[n]
            f.write(f'{n}    {n_max}    {n_min}    {n_avg}\n')

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

    get_best_metrics_id(report, kind=kind)
    extract_metrics(report)