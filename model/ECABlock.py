import torch
import torch.nn as nn
import math

class ECALayer(nn.Module):
    """ECA: Efficient Channel Attention (Pure Adaptive Version)"""
    def __init__(self, channel):
        super(ECALayer, self).__init__()
        
        # 1. 彻底自动化计算卷积核大小 k
        # 根据论文公式: k = psi(C) = |(log2(C) + b) / gamma|_odd
        gamma = 2
        b = 1
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 2. 1D 卷积，由自动计算的 k 确定感受野
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 压缩空间维度 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)

        # 2. 调整维度适配 Conv1d: [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)

        # 3. 跨通道局部交互 (Local Cross-channel Interaction)
        y = self.conv(y)

        # 4. 恢复维度 [B, C, 1, 1] 并生成注意力权重
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        # 5. 广播乘法
        return x * y.expand_as(x)