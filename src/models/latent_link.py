import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentLinkSynthesis(nn.Module):
    def __init__(self, dim: int, M: int = 50, delta: float = 0.5, S: int = 20):
        super().__init__()
        self.M = M
        self.delta = delta
        self.S = S
        self.scorer = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, H: torch.Tensor, z: torch.Tensor, A: torch.Tensor):
        """
        Returns augmented A and list of (i,j,confidence)
        """
        N = H.size(0)
        # top-M nodes
        top_idx = torch.topk(z, min(self.M, N)).indices
        pairs = []
        scores = []
        for i in top_idx:
            for j in top_idx:
                if i >= j: continue
                feat = torch.cat([H[i], H[j], z[i:i+1] + z[j:j+1]], dim=0)
                s = self.scorer(feat).squeeze()
                scores.append(s)
                pairs.append((i.item(), j.item()))
        if not scores:
            return A, []
        conf = torch.sigmoid(torch.stack(scores))
        # filter & top-S
        keep = (conf >= self.delta).nonzero(as_tuple=True)[0]
        if len(keep) > self.S:
            _, top = torch.topk(conf[keep], self.S)
            keep = keep[top]
        links = []
        A_aug = A.clone()
        for idx in keep:
            i, j = pairs[idx]
            c = conf[idx].item()
            A_aug[i, j, 0] = c          # add into first relation channel
            A_aug[j, i, 0] = c
            links.append((i, j, c))
        return A_aug, links
