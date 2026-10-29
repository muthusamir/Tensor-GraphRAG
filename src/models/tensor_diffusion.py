import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorDiffusion(nn.Module):
    def __init__(self, num_relations: int, steps: int = 3):
        super().__init__()
        self.steps = steps
        self.rel_weights = nn.Parameter(torch.ones(num_relations) / num_relations)
        self.betas = nn.Parameter(torch.ones(steps + 1) / (steps + 1))

    def forward(self, A: torch.Tensor, x0: torch.Tensor):
        """
        A: (N, N, R)
        x0: (N,)
        returns z: (N,)
        """
        x = x0
        states = [x]
        for k in range(self.steps):
            m = torch.zeros_like(x)
            for r in range(A.shape[-1]):
                m = m + self.rel_weights[r] * (A[:, :, r] @ x)
            x = F.normalize(torch.relu(m), p=1, dim=0)
            states.append(x)
        betas = F.softmax(self.betas, dim=0)
        z = sum(b * s for b, s in zip(betas, states))
        return z
