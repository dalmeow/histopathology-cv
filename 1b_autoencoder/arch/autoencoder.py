import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks (duplicated from 1a_unet/arch/unet.py to avoid sys.path issues)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x1, x2):
        x2 = self.up_scale(x2)
        dy = x1.size(2) - x2.size(2)
        dx = x1.size(3) - x2.size(3)
        x2 = F.pad(x2, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return torch.cat([x2, x1], dim=1)


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = _Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2): return self.conv(self.up(x1, x2))


class ResidualUpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up       = _Up(in_ch, out_ch)
        self.conv     = DoubleConv(in_ch, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x1, x2):
        a = self.up(x1, x2)
        return self.conv(a) + self.shortcut(a)


# ---------------------------------------------------------------------------
# Encoder  (identical to UNet encoder)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, base: int = 64, dropout: float = 0.3):
        super().__init__()
        b = base
        self.conv1 = DoubleConv(3,    b)
        self.down1 = DownLayer(b,     b*2)
        self.down2 = DownLayer(b*2,   b*4)
        self.down3 = DownLayer(b*4,   b*8)
        self.down4 = DownLayer(b*8,   b*16)
        self.drop  = nn.Dropout2d(p=dropout)
        self.x1 = self.x2 = self.x3 = self.x4 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x1 = self.conv1(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        x5      = self.drop(self.down4(self.x4))
        return x5


# ---------------------------------------------------------------------------
# ReconDecoder  (pre-training only — no skip connections)
# ---------------------------------------------------------------------------

class ReconDecoder(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        b = base
        self.up1 = nn.Sequential(nn.ConvTranspose2d(b*16, b*8, 2, 2), DoubleConv(b*8, b*8))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(b*8,  b*4, 2, 2), DoubleConv(b*4, b*4))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(b*4,  b*2, 2, 2), DoubleConv(b*2, b*2))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(b*2,  b,   2, 2), DoubleConv(b,   b))
        self.out = nn.Sequential(nn.Conv2d(b, 3, 1), nn.Sigmoid())

    def forward(self, x5: torch.Tensor) -> torch.Tensor:
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)


# ---------------------------------------------------------------------------
# SegDecoder  (fine-tuning — identical structure to UNet decoder)
# ---------------------------------------------------------------------------

class SegDecoder(nn.Module):
    def __init__(self, dimensions: int = 3, base: int = 64, dropout: float = 0.3, use_residual: bool = False):
        super().__init__()
        b = base
        UpBlock = ResidualUpLayer if use_residual else UpLayer
        self.up1 = UpBlock(b*16, b*8)
        self.up2 = UpBlock(b*8,  b*4)
        self.up3 = UpBlock(b*4,  b*2)
        self.up4 = UpBlock(b*2,  b)
        self.last_conv = nn.Conv2d(b, dimensions, 1)
        self.aux1 = nn.Conv2d(b*8, dimensions, 1)
        self.aux2 = nn.Conv2d(b*4, dimensions, 1)
        self.aux3 = nn.Conv2d(b*2, dimensions, 1)
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x5, x4, x3, x2, x1):
        x1_up = self.drop(self.up1(x4, x5))
        x2_up = self.drop(self.up2(x3, x1_up))
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        out   = self.last_conv(x4_up)
        if self.training:
            return out, self.aux1(x1_up), self.aux2(x2_up), self.aux3(x3_up)
        return out


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, mode: str = "pretrain", base: int = 64, dropout: float = 0.3,
                 use_residual: bool = False, dimensions: int = 3):
        super().__init__()
        assert mode in ("pretrain", "finetune"), f"mode must be 'pretrain' or 'finetune', got {mode!r}"
        self.mode          = mode
        self.encoder       = Encoder(base=base, dropout=dropout)
        self.recon_decoder = ReconDecoder(base=base)
        self.seg_decoder   = SegDecoder(dimensions=dimensions, base=base,
                                        dropout=dropout, use_residual=use_residual)

    def forward(self, x: torch.Tensor):
        x5 = self.encoder(x)
        if self.mode == "pretrain":
            return self.recon_decoder(x5)
        else:
            enc = self.encoder
            return self.seg_decoder(x5, enc.x4, enc.x3, enc.x2, enc.x1)
