import torch

from metrics import SammonError
from icecream import ic

class PSOFeatureSelection:
    def __init__(self, X, num_particles, fitness_function, w=0.6, c1=2, c2=2, n_metrics=10, device='cpu'):
        self.device = torch.device(device)
        self.num_particles = num_particles
        self.num_features = X.shape[1]
        self.max_features = n_metrics
        self.fitness_function = fitness_function(X, device=device)
        self.w = torch.tensor(w, device=self.device)
        self.c1 = torch.tensor(c1, device=self.device)
        self.c2 = torch.tensor(c2, device=self.device)

        # Initialize particles and velocities
        self.particles = torch.rand((self.num_particles, self.num_features), device=self.device)
        self.velocities = (2 * torch.rand((self.num_particles, self.num_features), device=self.device) - 1)

        # Initialize best positions and best global position
        self.pbest = self.particles
        self.pbest_fitness = self.compute_fitness_(self.pbest)
        # ic(self.pbest_fitness)
        
        self.gbest = self.pbest_fitness.argmin()
    
    def compute_fitness_(self, x: torch.tensor):
        _, indices = torch.topk(x, self.max_features)
        # ic(indices)
        fitness = self.fitness_function.compute(indices)
        return fitness

    def get_best_metrics(self):
        _, indices = torch.topk(self.pbest[self.gbest], self.max_features)
        return indices.cpu()

    def get_best_opt_value(self):
        return self.pbest_fitness[self.gbest].cpu()

    def step(self):
        # (6) Update velocities and positions
        # ic()
        r1, r2 = torch.rand(2, device=self.device)
        self.velocities = (self.w * self.velocities + 
                           self.c1 * r1 * (self.pbest - self.particles) +
                           self.c2 * r2 * (self.pbest[self.gbest, :] - self.particles) 
                          )
        self.particles = self.particles + self.velocities
        # (7) Evaluate on new particles
        # ic()
        fitness_new = self.compute_fitness_(self.particles)
        # ic()
        # (8) Update pbest
        fitness_mask = self.pbest_fitness >= fitness_new
    
        self.pbest[fitness_mask, :] = self.particles[fitness_mask, :]
        self.pbest_fitness[fitness_mask] = fitness_new[fitness_mask]
        # ic(fitness_new)
        # ic(self.pbest_fitness)

        # (8) Update gbest
        self.gbest = self.pbest_fitness.argmin()