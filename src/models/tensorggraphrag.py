import torch
import torch.nn as nn
from .encoder import SentenceEncoder
from .adjacency_tensor import LearnableAdjacencyTensor
from .tensor_diffusion import TensorDiffusion
from .spectral import SpectralContext
from .latent_link import LatentLinkSynthesis

class TensorGraphRAG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SentenceEncoder(config["encoder_name"])
        dim = config["embedding_dim"]
        self.adj = LearnableAdjacencyTensor(dim, config["num_relations"], config["top_k"])
        self.diffusion = TensorDiffusion(config["num_relations"], config["diffusion_steps"])
        self.spectral = SpectralContext(config["spectral_dim"])
        self.latent = LatentLinkSynthesis(dim, config["latent_m"], config["latent_delta"], config["latent_s"])
        self.tau = config["evidence_tau"]

    def forward(self, query: str, passages: list[str], return_details: bool = False):
        H = self.encoder(passages)                 # (N, d)
        q = self.encoder(query).squeeze(0)         # (d,)
        # initial activation
        sim = torch.matmul(H, q)
        x0 = torch.softmax(sim / 0.1, dim=0)
        # adjacency
        A = self.adj(H)                            # (N, N, R)
        # diffusion
        z = self.diffusion(A, x0)
        # spectral fusion
        z = self.spectral(z, A)
        # latent links
        A_aug, links = self.latent(H, z, A)
        # evidence selection
        mask = z >= self.tau
        evidence_idx = mask.nonzero(as_tuple=True)[0].tolist()
        if return_details:
            return evidence_idx, z, links
        return evidence_idx
