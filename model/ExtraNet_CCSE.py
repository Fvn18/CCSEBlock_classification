import torch
import torch.nn as nn
from .CCSEBlock import CrossChannelSegmentationExcitationBlock




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(att)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, ratio=1.5, use_spatial=False):
        super().__init__()
        mid_c = int(out_c * ratio)

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, padding=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.12),
            nn.Conv2d(mid_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.12),
        )

        self.ccse = CrossChannelSegmentationExcitationBlock(out_c, reduction=16)
                    
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial = SpatialAttention()

    def forward(self, x):
        out = self.conv_path(x) + self.shortcut(x)
        out = self.ccse(out)
        if self.use_spatial:
            out = self.spatial(out)
        return out


class ExtraNet_CCSE(nn.Module):
    def __init__(self, num_classes=7, use_spatial_attention=False, use_simple_fusion=True, input_channels=3):
        super(ExtraNet_CCSE, self).__init__()
        self.num_crops = 5
        self.use_simple_fusion = use_simple_fusion

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(0.05),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(0.05),
            nn.MaxPool2d(2),
        )

        self.res1 = ResidualBlock(128, 256,  ratio=1.5,  use_spatial=use_spatial_attention)
        self.res2 = ResidualBlock(256, 512,  ratio=1.5,  use_spatial=use_spatial_attention)
        self.res3 = ResidualBlock(512, 768,  ratio=1.4,  use_spatial=use_spatial_attention)
        self.res4 = ResidualBlock(768, 1024, ratio=1.25, use_spatial=use_spatial_attention)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        if not use_simple_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(num_classes * self.num_crops, 256),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )

    def forward_single(self, x):
        x = self.feature_extractor(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        if x.dim() == 5:
            B, n_crops, C, H, W = x.shape
            x = x.view(B * n_crops, C, H, W)
            out = self.forward_single(x)
            out = out.view(B, n_crops, -1)

            if self.use_simple_fusion:
                return out.mean(dim=1)
            else:
                return self.fusion(out.flatten(1))
        else:
            return self.forward_single(x)


if __name__ == "__main__":
    model = ExtraNet_CCSE(
        num_classes=7,
        use_spatial_attention=False,
        use_simple_fusion=True
    )

    print(model)
    
    x5 = torch.randn(4, 5, 1, 64, 64)
    out5 = model(x5)
    print("5-crop output shape:", out5.shape)

    x1 = torch.randn(4, 1, 64, 64)
    out1 = model(x1)
    print("Single crop output shape:", out1.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
