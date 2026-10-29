# Tensor GraphRAG

**Differentiable Multi-Hop Reasoning for Retrieval-Augmented Generation**

Official implementation of the paper:

> **Tensor GraphRAG: Differentiable Multi-Hop Reasoning for Retrieval-Augmented Generation**  
>

Tensor GraphRAG reformulates multi-hop retrieval-augmented generation as a **hybrid continuous–discrete** inference process over a learnable multi-relational document graph. It combines:

- Sparse **tensor diffusion** for query-conditioned multi-hop evidence propagation  
- **Spectral context modeling** for global graph organization  
- **Confidence-controlled latent-link synthesis** for bridging weakly connected evidence  

---

## Key Results

| Setting                        | Tensor GraphRAG | Strongest Baseline (HippoRAG 2) |
|--------------------------------|-----------------|--------------------------------|
| Overall EM                     | **63.7**        | 59.6                           |
| 4-hop EM                       | **54.8**        | 45.9                           |
| Relative degradation (1→4 hop) | **25.5%**       | 36.2%                          |
| Reasoning latency              | **89 ms**       | 105 ms                         |

---

## Repository Structure

```text
Tensor-GraphRAG/
├── configs/                  # Dataset-specific hyperparameters
├── src/
│   ├── data/                 # Preprocessing & hop-split scripts
│   ├── models/               # Core modules (encoder, diffusion, spectral, latent-link)
│   ├── train.py
│   └── evaluate.py
├── scripts/                  # One-click reproduction scripts
├── notebooks/                # Qualitative analysis
└── requirements.txt
