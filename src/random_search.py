import torch
from utils import OptimizationAlgorithm

class RandomSearch:
    """
    Random Search Algorithm for optimizing metrics subset.
    Performs single step() with n_trials guesses
    """
    def __init__(self, X, fitness_function, n_metrics=10, n_trials=900, **opt_params):
        self.num_features = X.shape[1]
        self.device = X.device
        self.max_features = n_metrics
        self.fitness_function = fitness_function(X)
        self.n_trials = n_trials
        self.best_fitness = None
        self.best_metrics = None

    def get_best_metrics(self):
        return self.best_metrics.cpu()

    def get_best_opt_value(self):
        return self.best_fitness.cpu()

    def step(self, i):
        indices = []
        for _ in range(self.n_trials):
            perm = torch.randperm(self.num_features)
            idx = perm[:self.max_features]
            indices.append(idx)
        
        fitness_vals = torch.tensor([self.fitness_function.compute(ind) for ind in indices], device=self.device)
        self.best_fitness = fitness_vals.min()
        self.best_metrics = indices[fitness_vals.argmin()]
