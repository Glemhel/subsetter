import torch
import numpy as np

class SimulatedAnnealing:
    def __init__(self, X, fitness_function, n_metrics=10, temp=10, **opt_params):
        self.num_features = X.shape[1]
        self.max_features = n_metrics
        self.device = X.device
        self.fitness_function = fitness_function(X)
        self.features_order = np.arange(X.shape[1])
        self.best_order = self.features_order
        self.best_fitness = self.compute_fitness_(self.features_order)
        self.temperature = temp
    
    def compute_fitness_(self, x: torch.tensor):
        indices = x[:self.max_features]
        fitness = self.fitness_function.compute(indices).cpu()
        return fitness

    def get_best_metrics(self):
        return self.best_order

    def get_best_opt_value(self):
        return self.best_fitness

    def step(self, i):
        
        # randomly swap two coordinates in the order: one from selected part, one from not
        i1 = np.random.randint(0, self.max_features)
        i2 = np.random.randint(self.max_features, self.num_features)
        self.features_order[[i1, i2]] = self.features_order[[i2, i1]]

        fitness_new = self.compute_fitness_(self.features_order)

        if fitness_new < self.best_fitness:
            self.best_fitness = fitness_new
            self.best_order = self.features_order.copy()

        diff = fitness_new - self.best_fitness
        t = self.temperature / (i + 1)
        metropolis = np.exp(-diff/t)
        if diff < 0 or np.random.rand() < metropolis:
            pass #  order is already swapped
        else:
            # swap back
            self.features_order[[i1, i2]] = self.features_order[[i2, i1]]

