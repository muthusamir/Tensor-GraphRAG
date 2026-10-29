import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableAdjacencyTensor(nn.Module):
    def __init__(self, dim: int, num_relations: int = 4, top_k: int = 32):
        super().__init__()
        self.num_relations = num_relations
        self.top_k = top_k
        self.W = nn.Parameter(torch.randn(num_relations, dim, dim) * 0.02)

    def forward(self, H: torch.Tensor):
        """
        H: (N, d)
        returns A: (N, N, R) sparse-friendly dense tensor (top-k masked)
        """
        N, d = H.shape
        A = []
        for r in range(self.num_relations):
            scores = H @ self.W[r] @ H.T          # (N, N)
            # top-k mask
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_idx, 1.0)
            scores = scores * mask
            A_r = F.softmax(scores / 0.1, dim=-1)  # temperature τ=0.1
            A.append(A_r)
        return torch.stack(A, dim=-1)              # (N, N, R)
