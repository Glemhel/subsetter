import torch
from utils import OptimizationAlgorithm

class PSOFeatureSelection:
    def __init__(self, X, fitness_function, w=0.6, c1=2, c2=2, n_metrics=10, num_particles=30, **opt_params):
        self.num_particles = num_particles
        self.num_features = X.shape[1]
        self.device = X.device
        self.max_features = n_metrics
        self.fitness_function = fitness_function(X)
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
        _, indices = torch.topk(x, self.max_features)
        fitness_vals = torch.tensor([self.fitness_function.compute(ind) for ind in indices], device=self.device)
        return fitness_vals

    def get_best_metrics(self):
        _, indices = torch.topk(self.pbest[self.gbest], self.num_features)
        return indices.cpu()

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
        # (7) Evaluate on new particles
        fitness_new = self.compute_fitness_(self.particles)
        # (8) Update pbest
        fitness_mask = self.pbest_fitness >= fitness_new
    
        self.pbest[fitness_mask, :] = self.particles[fitness_mask, :]
        self.pbest_fitness[fitness_mask] = fitness_new[fitness_mask]

        # (8) Update gbest
        self.gbest = self.pbest_fitness.argmin()
