import torch
import torch.nn.functional as F
from torch import nn

# Class weights derived from inverse pixel frequency (train set):
#   Tumor 69.7%  Stroma 25.1%  Other 5.3%
CLASS_WEIGHTS = torch.tensor([1/0.697, 1/0.251, 1/0.053])
CLASS_WEIGHTS = CLASS_WEIGHTS.sqrt()                  # soften: ~3.5x ratio instead of 13x
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum()   # normalise to sum to 1


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weights."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # passed to F.cross_entropy for class balancing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log-softmax for numerical stability
        log_p  = F.log_softmax(logits, dim=1)                  # (B, C, H, W)
        ce     = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")  # (B, H, W)
        # gather p_t = softmax probability of the true class
        p_t    = torch.exp(-ce)
        focal  = (1 - p_t) ** self.gamma * ce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs      = logits.softmax(dim=1)                              # (B, C, H, W)
        targets_oh = nn.functional.one_hot(targets, self.num_classes)   # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()            # (B, C, H, W)

        # Reduce over H, W only → (B, C); average over images then classes
        intersection = (probs * targets_oh).sum(dim=(2, 3))            # (B, C)
        cardinality  = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice_per_img = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice_per_img.mean()


def build_primary_criterion(loss_type: str, gamma: float, use_class_weights: bool, device: torch.device) -> nn.Module:
    """Return the primary (per-pixel) loss criterion."""
    weights = CLASS_WEIGHTS.to(device) if use_class_weights else None
    if loss_type == "focal":
        return FocalLoss(gamma=gamma, weight=weights)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}. Choose 'focal' or 'ce'.")
