import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # H x 1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 1 x W
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y); y = self.bn1(y); y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out = self.sigmoid(self.conv_h(x_h) + self.conv_w(x_w))
        return identity * out
