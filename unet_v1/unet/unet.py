import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


class UNet(nn.Module):
    def __init__(self, dimensions=3, dropout=0.3, base=96):
        super(UNet, self).__init__()
        b = base
        self.conv1 = DoubleConv(3,    b)
        self.down1 = DownLayer(b,     b*2)
        self.down2 = DownLayer(b*2,   b*4)
        self.down3 = DownLayer(b*4,   b*8)
        self.down4 = DownLayer(b*8,   b*16)
        self.up1 = UpLayer(b*16,  b*8)
        self.up2 = UpLayer(b*8,   b*4)
        self.up3 = UpLayer(b*4,   b*2)
        self.up4 = UpLayer(b*2,   b)
        self.last_conv  = nn.Conv2d(b,    dimensions, 1)
        # Auxiliary heads for deep supervision (used only during training)
        self.aux1 = nn.Conv2d(b*8,  dimensions, 1)
        self.aux2 = nn.Conv2d(b*4,  dimensions, 1)
        self.aux3 = nn.Conv2d(b*2,  dimensions, 1)
        # Dropout applied at bottleneck and first two decoder stages
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x1    = self.conv1(x)
        x2    = self.down1(x1)
        x3    = self.down2(x2)
        x4    = self.down3(x3)
        x5    = self.drop(self.down4(x4))   # bottleneck
        x1_up = self.drop(self.up1(x4, x5))
        x2_up = self.drop(self.up2(x3, x1_up))
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        out   = self.last_conv(x4_up)

        if self.training:
            # Return auxiliary outputs at up1, up2, up3 scales
            return out, self.aux1(x1_up), self.aux2(x2_up), self.aux3(x3_up)
        return out