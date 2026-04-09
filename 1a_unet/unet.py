import torch
import torch.nn.functional as F
from torch import nn


def _norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}. Choose 'batch' or 'instance'.")


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "instance"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            _norm(norm_type, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _norm(norm_type, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "instance"):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, norm_type)

    def forward(self, x):
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """Transposed-conv upsampling + skip-connection concatenation."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return torch.cat([x, skip], dim=1)


class UpLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "instance"):
        super().__init__()
        self.up   = _Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch, norm_type)

    def forward(self, skip, x):
        return self.conv(self.up(skip, x))


class ResidualUpLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "instance"):
        super().__init__()
        self.up       = _Up(in_ch, out_ch)
        self.conv     = DoubleConv(in_ch, out_ch, norm_type)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, skip, x):
        a = self.up(skip, x)
        return self.conv(a) + self.shortcut(a)


class UNet(nn.Module):
    """
    UNet with configurable normalisation, decoder type, deep supervision,
    and dropout.

    Args:
        num_classes:  number of output segmentation classes
        base_filters: number of filters in the first encoder block (doubles each level)
        norm_type:    'batch' or 'instance'
        use_residual: if True, decoder uses residual skip connections
        use_deep_sup: if True, returns auxiliary logits during training
        dropout_p:    Dropout2d probability at bottleneck and first two decoder stages
                      (0.0 = disabled)
    """
    def __init__(
        self,
        num_classes:  int   = 3,
        base_filters: int   = 64,
        norm_type:    str   = "instance",
        use_residual: bool  = False,
        use_deep_sup: bool  = False,
        dropout_p:    float = 0.0,
    ):
        super().__init__()
        b = base_filters
        UpBlock = ResidualUpLayer if use_residual else UpLayer
        self.use_deep_sup = use_deep_sup

        # Encoder
        self.enc1      = DoubleConv(3,    b,    norm_type)
        self.enc2      = DownLayer(b,     b*2,  norm_type)
        self.enc3      = DownLayer(b*2,   b*4,  norm_type)
        self.enc4      = DownLayer(b*4,   b*8,  norm_type)
        self.bottleneck = DownLayer(b*8,  b*16, norm_type)

        # Dropout at bottleneck + first two decoder stages
        self.drop = nn.Dropout2d(p=dropout_p)

        # Decoder
        self.dec1 = UpBlock(b*16, b*8,  norm_type)
        self.dec2 = UpBlock(b*8,  b*4,  norm_type)
        self.dec3 = UpBlock(b*4,  b*2,  norm_type)
        self.dec4 = UpBlock(b*2,  b,    norm_type)

        # Final head
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision (train only)
        if use_deep_sup:
            self.aux1 = nn.Conv2d(b*8, num_classes, kernel_size=1)
            self.aux2 = nn.Conv2d(b*4, num_classes, kernel_size=1)
            self.aux3 = nn.Conv2d(b*2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.drop(self.bottleneck(e4))

        d1 = self.drop(self.dec1(e4, bn))
        d2 = self.drop(self.dec2(e3, d1))
        d3 = self.dec3(e2, d2)
        d4 = self.dec4(e1, d3)
        out = self.head(d4)

        if self.use_deep_sup and self.training:
            sz = x.shape[-2:]
            up = lambda t: F.interpolate(t, size=sz, mode="bilinear", align_corners=False)
            return out, up(self.aux1(d1)), up(self.aux2(d2)), up(self.aux3(d3))
        return out
