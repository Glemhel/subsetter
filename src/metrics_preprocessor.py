import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

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

    with open('analysis/url_encoder.json', 'w') as f:
        json.dump(repo_url_encoder, f)

    function_metrics = d.explode(function_metrics_list)
    function_metrics.replace({'url': repo_url_encoder}, inplace=True)
    
    scaler = StandardScaler()
    function_metrics[function_metrics_list] = scaler.fit_transform(function_metrics[function_metrics_list])
    function_metrics.to_csv(args.output, index=False)