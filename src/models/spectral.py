import torch
import torch.nn as nn
from scipy.sparse.linalg import eigsh
import numpy as np

class SpectralContext(nn.Module):
    def __init__(self, spectral_dim: int = 64, mix: float = 0.7):
        super().__init__()
        self.p = spectral_dim
        self.mix = mix
        self.proj = nn.Linear(spectral_dim, 1)

    @torch.no_grad()
    def build_laplacian(self, A_agg: torch.Tensor):
        """A_agg: (N, N) symmetrized aggregated adjacency"""
        A = (A_agg + A_agg.T) / 2
        A = A.cpu().numpy()
        deg = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        # partial eigen-decomposition
        vals, vecs = eigsh(L, k=self.p, which="SM")
        return torch.from_numpy(vecs).float()

    def forward(self, z: torch.Tensor, A: torch.Tensor, q: torch.Tensor = None):
        A_agg = A.mean(dim=-1)                     # simple aggregation
        U = self.build_laplacian(A_agg).to(z.device)  # (N, p)
        # query-conditioned spectral score (optional)
        spectral_score = torch.softmax(U @ U.T @ z, dim=0)
        return self.mix * z + (1 - self.mix) * spectral_score
