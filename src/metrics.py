import torch


class Metrics:
    def __init__(self):
        pass

    def compute(self, X, d_original=None, excluded_columns=None):
        raise NotImplementedError("Subclasses must implement the compute method")


class SammonError(Metrics):
    def __init__(self, X):
        self.X = X
        self.device = X.device
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


class KruskalStress(Metrics):
    def __init__(self, X):
        self.X = X
        self.device = X.device
        self.n_classes = X.shape[0]
        self.n_metrics = X.shape[1]
        self.d_original = torch.cdist(X, X, p=2) + 1e-15
        self.error_denominator = (self.d_original.triu(diagonal=1) ** 2).sum() ** 0.5

    def compute(self, included_columns=[None]):
        # Create a mask to exclude certain columns
        mask = torch.zeros(self.n_metrics, device=self.device)
        mask[included_columns] = 1

        X_masked = self.X * mask

        # distance graph matrix
        d_subset = torch.cdist(X_masked, X_masked, p=2)

        # filtering edges
        n_ban = int(d_subset.shape[0] * 0.9) - 1
        n_left = d_subset.shape[0] - n_ban
        _, ind = d_subset.topk(n_ban, largest=True)
        d_subset.scatter_(index=ind, dim=1, value=0)

        # Floyd-Warshall
        n = d_subset.shape[0]
        for k in range(n):
            n_n = d_subset[:, k].contiguous().view(n, 1) + d_subset[k, :].view(1, n)
            d_subset = torch.min(d_subset, n_n)

        d_diff = (self.d_original - d_subset) ** 2
        e = d_diff.triu(diagonal=1).sum() ** 0.5
        error = e / self.error_denominator
        return error


class EditDistance(Metrics):
    """
    Edit distance between metrics subspaces.
    On a matrix D of pairwise distances between each pair of objects(functions/structs),
    we filter out portion of edges to make an undirected, non-weigthed graph D'.
    We then compute edit distance between original graph D' and subspaced graph D'', as simple
    sum of absolute differences between adjacency matrices.
    """

    def __init__(self, X):
        self.X = X
        self.device = X.device
        self.n_classes = X.shape[0]
        self.n_metrics = X.shape[1]
        self.d_original = self.make_adj_matrix_(X)

    def make_adj_matrix_(self, X):
        # distance graph
        adj_matrix = torch.cdist(X, X, p=2)
        # filtering edges - largest 70%
        n_ban = int(adj_matrix.shape[0] * 0.7) - 1
        _, ind = adj_matrix.topk(n_ban, largest=True)
        adj_matrix_01 = torch.ones_like(adj_matrix)
        adj_matrix_01.scatter_(index=ind, dim=1, value=0)
        return adj_matrix_01

    def compute(self, included_columns=None):
        # Create a mask to exclude certain columns
        mask = torch.zeros(self.n_metrics, device=self.device)
        mask[included_columns] = 1

        X_masked = self.X * mask

        # distance graph matrix
        d_subset = self.make_adj_matrix_(X_masked)

        edit_dist = (d_subset - self.d_original).abs().sum()
        value = edit_dist / d_subset.shape[0] ** 2
        return value
