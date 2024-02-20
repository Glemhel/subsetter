import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

def list_uncorrelated_features(df, thr = 0.9):
    """
    Return list of indices of columns that are not higly correlated
    """
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation not greater than thr
    uncorrelated_features_list = [i for i, column in enumerate(upper.columns) if not(any(upper[column] > thr))]

    return uncorrelated_features_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Metrics preprocessor",
        description="Takes the raw metrics and transforms them into a format suitable for the processing"
    )
    parser.add_argument('input')
    parser.add_argument('metrics_list')
    parser.add_argument('output')
    args = parser.parse_args()
    
    d = pd.read_json(args.input, dtype=True)
    d = d.join(pd.json_normalize(d.pop('meta')))
    d = d.join(pd.json_normalize(d.pop('metrics')))
    d.drop(columns='commit', inplace=True)

    with open(args.metrics_list) as f:
        function_metrics_list = [l.strip() for l in f]

    d = d[function_metrics_list + ['url']]

    unique_urls = d.url.unique()
    repo_url_encoder = dict([(url, i) for i, url in enumerate(unique_urls)])

    with open('../analysis/url_encoder.json', 'w') as f:
        json.dump(repo_url_encoder, f)

    function_metrics = d.explode(function_metrics_list)
    function_metrics.replace({'url': repo_url_encoder}, inplace=True)
    
    # Drop NaNs report
    nans_per_record = function_metrics.isna().sum(axis=1)
    print(f'In total, there are {(nans_per_record > 0).sum()} records containing at least 1 NaN value')
    print('Detailed report for each repository with a non-zero number of NaNs')
    for repo in unique_urls:
        repo_id = repo_url_encoder[repo]
        repo_records = function_metrics[function_metrics['url'] == repo_id]
        n_records = repo_records.shape[0]
        nan_records = ((repo_records.isna().sum(axis=1) > 0).sum())
        if nan_records > 0:
            print(f'{repo}({repo_id}) contains {nan_records} (out of {n_records}) records with NaNs')
    
    function_metrics.dropna(inplace=True)
    
    scaler = StandardScaler()
    function_metrics[function_metrics_list] = scaler.fit_transform(function_metrics[function_metrics_list])
    # Drop highly correlated features
    # XXX: url column is considered but not dropped due to no correlation! It would be bad if it's dropped..
    print("Uncorrelated features list:")
    print(list_uncorrelated_features(function_metrics, thr=0.9))
    function_metrics.to_csv(args.output, index=False)
