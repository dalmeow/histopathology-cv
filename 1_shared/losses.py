import torch
import torch.nn.functional as F
from torch import nn

# Class frequencies from training set: Tumor 69.7%, Stroma 25.1%, Other 5.3%
_CLASS_FREQ = torch.tensor([0.697, 0.251, 0.053])


def _class_weights(device: torch.device) -> torch.Tensor:
    """Sqrt-softened inverse-frequency weights, normalised to sum to 1."""
    w = 1.0 / _CLASS_FREQ.sqrt()
    w = w / w.sum()
    return w.to(device)


# Losses

class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weights."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        p_t   = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    """Per-image Dice loss averaged over batch and classes."""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs      = logits.softmax(dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        inter      = (probs * targets_oh).sum(dim=(2, 3))
        card       = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice       = (2 * inter + self.eps) / (card + self.eps)
        return 1 - dice.mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient for a sorted binary ground-truth vector."""
    p    = len(gt_sorted)
    gts  = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard      = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss (Berman et al. 2018), batch mode.

    Operates on softmax probabilities. Flattens the entire batch before
    computing per-class Lovász gradients, giving stable signals even for
    rare classes (e.g. Other at 5.3% of pixels).

    Only averages over classes present in the batch ('present' mode).
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)           # (B, C, H, W)
        B, C, H, W = probs.shape

        # Flatten batch → (N, C) and (N,)
        probs_flat   = probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        losses = []
        for c in range(C):
            fg = (targets_flat == c).float()
            if fg.sum() == 0:               # class absent in batch — skip
                continue
            errors = (fg - probs_flat[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

        if not losses:
            return probs.sum() * 0   # no class present — zero loss, graph intact
        return torch.stack(losses).mean()


# Criterion builder

def build_criterion(loss_type: str, use_class_weights: bool, loss_lambda: float,
                    device: torch.device):
    """
    Returns a callable  criterion(logits, targets) -> scalar loss.

    loss_type:
        "ce"           — CrossEntropyLoss (+ optional class weights)
        "ce+dice"      — weighted CE + lambda * per-image DiceLoss
        "focal+lovasz" — FocalLoss(gamma=2, unweighted) + lambda * LovaszSoftmaxLoss
    """
    weights = _class_weights(device) if use_class_weights else None

    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == "ce+dice":
        ce   = nn.CrossEntropyLoss(weight=weights)
        dice = DiceLoss(num_classes=3)
        def criterion(logits, targets):
            return ce(logits, targets) + loss_lambda * dice(logits, targets)
        return criterion

    elif loss_type == "focal+lovasz":
        focal  = FocalLoss(gamma=2.0, weight=None)
        lovasz = LovaszSoftmaxLoss(num_classes=3)
        def criterion(logits, targets):
            return focal(logits, targets) + loss_lambda * lovasz(logits, targets)
        return criterion

    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. "
                         f"Choose 'ce', 'ce+dice', or 'focal+lovasz'.")
