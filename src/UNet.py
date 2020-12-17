"""
Implementation of U-Net model with batch normalization and optional dropout
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """ Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DownBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.DoubleConv = DoubleConv(in_channels, out_channels)
        self.dropout = dropout
        self.dropoutlayer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.DoubleConv(x)
        if self.dropout > 0:
            x = self.dropoutlayer(x)
        return x


class UpBlock(nn.Module):
    """Upscaling * 2 then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.dropout = dropout
        self.dropoutlayer = nn.Dropout(p=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        if self.dropout > 0:
            x = self.dropoutlayer(x)
        return x


class UNet(nn.Module):
    """
    U-net model
    """

    def __init__(self, n_channels=3, n_classes=1, n_filters=64, dropout=0.2):
        super(UNet, self).__init__()
        nf = n_filters
        self.inc = DoubleConv(n_channels, nf)
        self.down1 = DownBlock(nf, 2 * nf, dropout)
        self.down2 = DownBlock(2 * nf, 4 * nf, dropout)
        self.down3 = DownBlock(4 * nf, 8 * nf, dropout)
        self.down4 = DownBlock(8 * nf, 16 * nf, dropout)
        self.up1 = UpBlock(16 * nf, 8 * nf, dropout)
        self.up2 = UpBlock(8 * nf, 4 * nf, dropout)
        self.up3 = UpBlock(4 * nf, 2 * nf, dropout)
        self.up4 = UpBlock(2 * nf, nf, dropout)
        self.OutConv = nn.Conv2d(nf, n_classes, kernel_size=1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.OutConv(x)
        # x = self.softmax(x)
        return x
