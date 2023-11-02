import yaml
import argparse
from pathlib import Path
from math import isnan

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

    print("Extracting metrics")
    transformed = {}
    for repo_id in report.keys():
        for n in report[repo_id].keys():
            optimums = report[repo_id][n]['optimums']
            cur_min, cur_max, cur_avg = min(optimums), max(optimums), sum(optimums) / len(optimums)
            if n in transformed.keys():
                stored_min, stored_max, stored_avg = transformed[n]
                stored_min = min(cur_min, stored_min) if not isnan(cur_min) else stored_min
                stored_max = max(cur_max, stored_max) if not isnan(cur_max) else stored_max
                stored_avg += cur_avg if not isnan(cur_avg) else 0
                transformed[n] = [stored_min, stored_max, stored_avg]
            else:
                transformed[n] = [cur_min, cur_max, cur_avg]

    # Normalize averages
    n_repos = len(report.keys())
    for n in transformed.keys():
        transformed[n][2] /= n_repos
   
    print("Saving transformed report")
    with open(f"{Path(args.reportfile).stem}.dat", "w") as f:
        # Write header
        f.write("n_metrics    max    min   avg\n")
        for n in transformed.keys():
            n_max, n_min, n_avg = transformed[n]
            f.write(f'{n}    {n_max}    {n_min}    {n_avg}\n')
