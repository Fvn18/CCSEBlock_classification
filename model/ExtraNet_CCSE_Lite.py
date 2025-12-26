import torch
import torch.nn as nn
import torch.nn.functional as F
from .CCSEBlock import CrossChannelSegmentationExcitationBlock

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size, padding=padding, 
                                   stride=stride, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class LiteResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio=3, stride=1):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_c * expand_ratio))
        self.use_res_connect = (self.stride == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_c, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            CrossChannelSegmentationExcitationBlock(hidden_dim, reduction=8),

            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ExtraNet_Lite_CCSE(nn.Module):
    def __init__(self, num_classes=7, use_simple_fusion=True, input_channels=3):
        super(ExtraNet_Lite_CCSE, self).__init__()
        self.use_simple_fusion = use_simple_fusion
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            LiteResidualBlock(64, 64, stride=1),
            LiteResidualBlock(64, 64, stride=1)
        )
        
        self.layer2 = nn.Sequential(
            LiteResidualBlock(64, 128, stride=2),
            LiteResidualBlock(128, 128, stride=1),
            LiteResidualBlock(128, 128, stride=1)
        )

        self.layer3 = nn.Sequential(
            LiteResidualBlock(128, 256, stride=2),
            LiteResidualBlock(256, 256, stride=1),
            LiteResidualBlock(256, 256, stride=1),
            LiteResidualBlock(256, 256, stride=1)
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_single(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        if x.dim() == 5: 
            B, n_crops, C, H, W = x.shape
            x = x.view(B * n_crops, C, H, W)
            out = self.forward_single(x)
            out = out.view(B, n_crops, -1)
            return out.mean(dim=1)
        else:
            return self.forward_single(x)
