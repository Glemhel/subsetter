import torch

class Metrics:
    def __init__(self):
        pass

    def compute(self, X, d_original=None, excluded_columns=None):
        raise NotImplementedError("Subclasses must implement the compute method")


class SammonError(Metrics):
    def __init__(self, X, device='cpu'):
        self.X = X
        self.device = torch.device(device)
        self.n_classes = X.shape[0]
        self.n_metrics = X.shape[1]
        self.d_original = torch.cdist(X, X, p=2) + 1e-15
        self.error_denominator = self.d_original.triu(diagonal=1).sum()


    def compute(self, included_columns=None):
        # Create a mask to exclude certain columns
        mask = torch.zeros(self.n_metrics, device=self.device)
        mask[included_columns] = 1
        # mask deselected to 0
        X_masked = self.X * mask
        # compute error
        d_subset = torch.cdist(X_masked, X_masked, p=2)
        d_diff = (self.d_original - d_subset) ** 2 / self.d_original
        e = d_diff.triu(diagonal=1).sum()
        sammon_error = e / self.error_denominator
        return sammon_error
