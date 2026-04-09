"""
model.py — 2b_contrastive

Imports NucleiResNet and ResNet18Encoder from 2a_classifier (single source
of truth for encoder architectures).

Adds:
  SimCLRProjectionHead — MLP projection head used only during pre-training.
  NTXentLoss           — NT-Xent contrastive loss (SimCLR, Chen et al. 2020).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "2a_classifier"))

# Re-export encoders — all encoder architecture lives in 2a_classifier/model.py
from model import NucleiResNet, ResNet18Encoder  # noqa: F401


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class SimCLRProjectionHead(nn.Module):
    """
    MLP projection head: Linear(in_dim→hid_dim, no bias) → BN → ReLU → Linear(hid_dim→out_dim).
    Discarded after pre-training — only the encoder weights are kept.
    """

    def __init__(self, in_dim: int, hid_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss (SimCLR, Chen et al. 2020).

    For a batch of N patches → 2N views (view1, view2).
    Positive pair for view i: the other view of the same patch.
    All other 2(N-1) views in the batch are negatives.

    Loss = -log[ exp(sim(zi,zj)/τ) / Σ_{k≠i} exp(sim(zi,zk)/τ) ]
    averaged over all 2N views.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : (N, D) raw projection embeddings (L2-normalised inside)
        """
        N = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2N, D)
        sim = torch.mm(z, z.T) / self.temperature              # (2N, 2N)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices: view i → view i+N and vice versa
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])  # (2N,)

        return F.cross_entropy(sim, labels)
