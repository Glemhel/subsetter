from pso import PSOFeatureSelection
from metrics import SammonError
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import pickle
import gc
import os

from icecream import ic

"""
Analyze repository
"""
def subsetter(X: torch.tensor, method, opt_function, max_iterations=30, n_metrics=10):
    num_particles = 30
    optimizer = method(X, num_particles, opt_function, n_metrics=n_metrics)
    for i in range(max_iterations):
        optimizer.step()

    return optimizer.get_best_metrics(), optimizer.get_best_opt_value()



def get_data_i(data, i, kind='function'):
    datai = data[data['repo_id'] == i]
    return torch.tensor(datai.drop(columns=['repo_id']).values)


def run_analysis(n_metrics=10):

    data = pd.read_csv(os.path.join('..', 'data', 'function_metrics.csv'), dtype=np.float32)
    data.rename(columns={'Unnamed: 0': 'repo_id'}, inplace=True)
    data.repo_id = data.repo_id.astype(int)
    repos_order_by_size = data.repo_id.value_counts().index.values
    opt_function = SammonError
    method = PSOFeatureSelection
    metrics_vote = np.zeros(data.shape[1])
    opt_values = []
    # for i in range(10, 11):
    for i in tqdm(range(30, 181)):
        repo_data = get_data_i(data, repos_order_by_size[i])
        device = torch.device('cuda') 
        repo_data_cuda = repo_data.to(device)
        selected_metrics, opt_value = subsetter(repo_data_cuda, method, opt_function, n_metrics=n_metrics)
        metrics_vote[selected_metrics] += 1
        opt_values.append(opt_value)
        torch.cuda.empty_cache()
        gc.collect()

    print('Metrics votecount: ')
    print(metrics_vote)
    print('Optimal values: ')
    print(opt_values)

    return metrics_vote, opt_values


if __name__ == '__main__':
    res = {}
    torch.manual_seed(42)
    for n_metrics in [2, 3, 4, 5, 10, 15, 20, 30]:
        votes, opts = run_analysis(n_metrics)
        res1 = (votes, opts)
        with open(os.path.join('..', 'analysis', f'samplerun-n_metrics-{n_metrics}'), 'wb') as handle:
            b = pickle.dump(res1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res[n_metrics] = res1
    with open(os.path.join('..', 'analysis', f'samplerun-all'), 'wb') as handle:
        b = pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
