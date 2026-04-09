"""
Model definitions for Task 2a: End-to-end nuclei classification.

Three architectures (additive ablation):
  SimpleClassifier  (~94K params)  — Exp 1: naive baseline
  NucleiResNet      (~2.8M params) — Exp 2–4: residual stages + optional ECA
  ResNet18Encoder   (~11M params)  — Exp 5: ImageNet pretrained backbone
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 3
NUM_CLASSES  = 3


# ===========================================================================
# SimpleClassifier — Exp 1
# ===========================================================================

class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleClassifier(nn.Module):
    """
    Three-block CNN: Conv→BN→ReLU→MaxPool ×3, GAP, Linear(128→3).
    Input  : (B, 3, 100, 100)
    Output : (B, 3) logits
    """

    def __init__(self, in_channels: int = IN_CHANNELS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feature_dim = 128
        self.conv1 = _ConvBlock(in_channels, 32)
        self.conv2 = _ConvBlock(32, 64)
        self.conv3 = _ConvBlock(64, 128)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pool(x).flatten(1)   # (B, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# NucleiResNet — Exp 2 (base), Exp 3 (+ECA), Exp 4 (+Mixup)
# ===========================================================================

class _ECA(nn.Module):
    """
    Efficient Channel Attention (Wang et al., 2020).
    Kernel size derived from channels: k = nearest odd to |log2(c)/2 + 0.5|
    Near-zero parameter overhead (~k weights).
    """

    def __init__(self, channels: int):
        super().__init__()
        k = int(abs(math.log2(channels) / 2 + 0.5))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)                       # (B, 1, C)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class _ResBlock(nn.Module):
    """
    Conv-BN-ReLU-Conv-BN + optional ECA + shortcut (1×1 if dims change).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_eca: bool = False):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.eca      = _ECA(out_ch) if use_eca else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.eca(self.bn2(self.conv2(out)))
        return F.relu(out + self.shortcut(x), inplace=True)


class NucleiResNet(nn.Module):
    """
    Three-stage residual network for 100×100 nucleus patches.

    Stem  : Conv(3→64, 7×7 stride2) → BN → ReLU → MaxPool(3, stride2)
                                                     → (B, 64, 25, 25)
    Stage1: 2× ResBlock(64→64)                      → (B, 64, 25, 25)
    Stage2: 2× ResBlock(64→128, stride2)            → (B, 128, 13, 13)
    Stage3: 2× ResBlock(128→256, stride2)           → (B, 256,  7,  7)
    GAP → Dropout(0.3) → Linear(256→3)
    """

    def __init__(
        self,
        in_channels: int   = IN_CHANNELS,
        num_classes:  int   = NUM_CLASSES,
        dropout:      float = 0.3,
        use_eca:      bool  = False,
    ):
        super().__init__()
        self.feature_dim = 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(64,  64,  n_blocks=2, stride=1, use_eca=use_eca)
        self.stage2 = self._make_stage(64,  128, n_blocks=2, stride=2, use_eca=use_eca)
        self.stage3 = self._make_stage(128, 256, n_blocks=2, stride=2, use_eca=use_eca)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head    = nn.Linear(256, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride, use_eca=False):
        blocks = [_ResBlock(in_ch, out_ch, stride=stride, use_eca=use_eca)]
        for _ in range(1, n_blocks):
            blocks.append(_ResBlock(out_ch, out_ch, stride=1, use_eca=use_eca))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.pool(x).flatten(1)   # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.get_features(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# ResNet18Encoder — Exp 5
# ===========================================================================

class ResNet18Encoder(nn.Module):
    """
    ImageNet-pretrained ResNet-18 with FC replaced by a task-specific head.
    feature_dim = 512 (AdaptiveAvgPool output before head).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feature_dim = 512
        import torchvision.models as tv
        base = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Linear(512, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)   # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
