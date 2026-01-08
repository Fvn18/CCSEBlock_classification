import torch
import torch.nn as nn

class CrossChannelSegmentationExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channels = channels
        self.even_channels = channels // 2 * 2
        self.half = self.even_channels // 2

        mid1 = max(self.half // reduction, 1)
        mid2 = max(self.even_channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.half, mid1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid1, self.half, bias=False),
            nn.Sigmoid()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.even_channels, mid2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid2, self.half, bias=False),
            nn.Sigmoid()
        )

        if channels % 2 == 1:
            self.single_mlp = nn.Sequential(
                nn.Linear(1, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, h, w = x.shape

        x_even = x[:, :self.even_channels, :, :]
        x1 = x_even[:, 0::2, :, :]
        x2 = x_even[:, 1::2, :, :]

        z1 = self.avg_pool(x1).view(b, self.half)
        z2 = self.avg_pool(x2).view(b, self.half)

        w1 = self.mlp1(z1).view(b, self.half, 1, 1)
        z_cat = torch.cat([z1, z2], dim=1)
        w2 = self.mlp2(z_cat).view(b, self.half, 1, 1)

        out = torch.empty_like(x_even)
        out[:, 0::2, :, :] = x1 * w1
        out[:, 1::2, :, :] = x2 * w2

        if self.channels % 2 == 0:
            return out

        x_last = x[:, -1:, :, :]
        z_last = self.avg_pool(x_last).view(b, 1)
        w_last = self.single_mlp(z_last).view(b, 1, 1, 1)
        x_last = x_last * w_last

        return torch.cat([out, x_last], dim=1)

