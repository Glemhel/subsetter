import torch
from utils import OptimizationAlgorithm

class PSOFeatureSelection:
    """
    Particle swarm optimization algorithm, which optimise a number of vectors in
    given metric, using each others values. 
    """
    def __init__(self, X, fitness_function, w=0.6, c1=2, c2=2, n_metrics=10, num_particles=30, pso_selection_policy='topk-max', metrics_subset_list=None ,**opt_params):
        self.pso_selection_policy = pso_selection_policy
        self.num_particles = num_particles
        if metrics_subset_list is None:
            metrics_subset_list = X.columns # should not work...
        else:
            pass
            # all_present = min([metric in X.columns for metric in metrics_subset_list])
            # if not all_present:
            #     raise ValueError("Some of metrics from metrics_subset_list are not present in dataset")
        # preprocess to take only features from metrics_subset_list
        X_subset = X[:, metrics_subset_list]

        self.num_features = X_subset.shape[1]
        self.device = X_subset.device
        self.max_features = n_metrics
        self.fitness_function = fitness_function(X, metrics_subset_list=metrics_subset_list)
        self.w = torch.tensor(w, device=self.device)
        self.c1 = torch.tensor(c1, device=self.device)
        self.c2 = torch.tensor(c2, device=self.device)

        # Initialize particles and velocities
        self.particles = torch.rand((self.num_particles, self.num_features), device=self.device)
        self.velocities = (2 * torch.rand((self.num_particles, self.num_features), device=self.device) - 1)

        # Initialize best positions and best global position
        self.pbest = self.particles
        self.pbest_fitness = self.compute_fitness_(self.pbest)
        
        self.gbest = self.pbest_fitness.argmin()

    def compute_fitness_(self, x: torch.tensor):
        if self.pso_selection_policy == 'topk-max':
            _, indices = torch.topk(x, self.max_features, largest=True)
        elif self.pso_selection_policy == 'topk-min':
            _, indices = torch.topk(x, self.max_features, largest=False)
        elif self.pso_selection_policy == 'rand':
            indices = []
            for i in range(self.num_particles):
                perm = torch.randperm(self.num_features)
                idx = perm[:self.max_features]
                indices.append(idx)
            # indices = [torch.randint(low=0, high=self.num_features, size=(self.max_features,)) for _ in range(self.num_particles)]
            # _, indices = torch.topk(x, self.max_features, largest=True)
        
        fitness_vals = torch.tensor([self.fitness_function.compute(ind) for ind in indices], device=self.device)
        return fitness_vals

    def get_best_metrics(self):
        if self.pso_selection_policy == 'topk-max':
            _, indices = torch.topk(self.pbest[self.gbest], self.num_features, largest=True)
            return indices.cpu()
        elif self.pso_selection_policy == 'topk-min':
            _, indices = torch.topk(self.pbest[self.gbest], self.num_features, largest=False)
            return indices.cpu()
        elif self.pso_selection_policy == 'rand':
            return torch.tensor([])

    def get_best_opt_value(self):
        return self.pbest_fitness[self.gbest].cpu()

    def step(self, i):
        # (6) Update velocities and positions
        r1, r2 = torch.rand(2, device=self.device)
        self.velocities = (self.w * self.velocities + 
                           self.c1 * r1 * (self.pbest - self.particles) +
                           self.c2 * r2 * (self.pbest[self.gbest, :] - self.particles) 
                          )
        self.particles = self.particles + self.velocities
        self.particles = torch.clamp(self.particles, 0, 1)
        # (7) Evaluate on new particles
        fitness_new = self.compute_fitness_(self.particles)
        # (8) Update pbest
        fitness_mask = self.pbest_fitness >= fitness_new
    
        self.pbest[fitness_mask, :] = self.particles[fitness_mask, :]
        self.pbest_fitness[fitness_mask] = fitness_new[fitness_mask]

        # (8) Update gbest
        self.gbest = self.pbest_fitness.argmin()
