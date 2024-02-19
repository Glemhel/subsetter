import torch
import itertools
from utils import OptimizationAlgorithm

class ExactSearch:
    """
    Exact Search Algorithm for optimizing metrics subset.
    Enumerates all possible subsets of metrics and finds minimum
    """
    def __init__(self, X, fitness_function, n_metrics=10, **opt_params):
        raise NotImplementedError()
        self.num_features = X.shape[1]
        self.device = X.device
        self.max_features = n_metrics
        self.fitness_function = fitness_function(X)
        self.best_fitness = None
        self.best_metrics = None

        if n_metrics >= 13:
            raise ValueError("Too many options to enumeratre, aborting...")

    def get_best_metrics(self):
        return self.best_metrics.cpu()

    def get_best_opt_value(self):
        return self.best_fitness.cpu()

    def step(self, i):
        raise NotImplementedError()
        subset_to_loss = []
        for subset in itertools.combinations(list(range(self.num_features)), self.max_features):
            # Create a binary mask tensor for the current subset
            mask_tensor = torch.zeros(self.num_features, dtype=torch.bool, device='cuda')
            mask_tensor[[idx for idx in subset]] = 1
            loss = sammon.compute(mask_tensor)
            subset_to_loss.append((list(subset), loss))

        subset_to_loss.sort(key=lambda x: x[1].item())
        indices = []
        for _ in range(self.n_trials):
            perm = torch.randperm(self.num_features)
            idx = perm[:self.max_features]
            indices.append(idx)
        
        fitness_vals = torch.tensor([self.fitness_function.compute(ind) for ind in indices], device=self.device)
        self.best_fitness = fitness_vals.min()
        self.best_metrics = indices[fitness_vals.argmin()]
