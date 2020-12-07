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


class down_block(nn.Module):
    """ Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=True):
        super(down_block, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.DoubleConv = DoubleConv(in_channels, out_channels)
        self.dropout = dropout
        self.dropoutlayer = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.DoubleConv(x)
        if self.dropout:
            x = self.dropoutlayer(x)
        return x


class up_block(nn.Module):
    """Upscaling * 2 then double conv"""

    def __init__(self, in_channels, out_channels, dropout=True):
        super(up_block, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels + in_channels // 2, out_channels, in_channels // 2)
        self.dropout = dropout
        self.dropoutlayer = nn.Dropout(p=0.2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        if self.dropout:
            x = self.dropoutlayer(x)
        return x


class UNet(nn.Module):
    """
    U-net model
    """

    def __init__(self, n_channels=3, n_classes=1, n_filters=64):
        super(UNet, self).__init__()
        N = n_filters
        self.inc = DoubleConv(n_channels, N)
        self.down1 = down_block(N, 2 * N)
        self.down2 = down_block(2 * N, 4 * N)
        self.down3 = down_block(4 * N, 8 * N)
        self.down4 = down_block(8 * N, 16 * N)
        self.up1 = up_block(16 * N, 8 * N)
        self.up2 = up_block(8 * N, 4 * N)
        self.up3 = up_block(4 * N, 2 * N)
        self.up4 = up_block(2 * N, N)
        self.OutConv = nn.Conv2d(N, n_classes, kernel_size=1)
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
