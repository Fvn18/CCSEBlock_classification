import torch
import torch.nn as nn


class ECAAttention(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECAAttention, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(cat)
        return x * self.sigmoid(att)


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=1.5, use_spatial_att=False):
        super().__init__()
        hidden_channels = int(out_channels * ratio)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.12),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.12),
        )

        self.eca = ECAAttention(out_channels, k_size=3)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        out = x1 + x2
        out = self.eca(out)
        if self.use_spatial_att:
            out = self.spatial(out)
        return out


class CNNmodel(nn.Module):
    def __init__(self, image_size=64, num_classes=7, use_spatial_attention=False, use_simple_fusion=True, input_channels=3):
        super(CNNmodel, self).__init__()
        self.image_size = image_size
        self.num_crops = 5
        self.use_simple_fusion = use_simple_fusion

        self.extract = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.first_residual = residual_block(128, 256, ratio=1.5, use_spatial_att=use_spatial_attention)
        self.second_residual = residual_block(256, 512, ratio=1.5, use_spatial_att=use_spatial_attention)
        self.third_residual = residual_block(512, 768, ratio=1.4, use_spatial_att=use_spatial_attention)
        self.fourth_residual = residual_block(768, 1024, ratio=1.25, use_spatial_att=use_spatial_attention)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

        if not use_simple_fusion:
            self.crop_fusion = nn.Sequential(
                nn.Linear(num_classes * self.num_crops, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(128, num_classes)
            )

    def forward_single_crop(self, x):
        x = self.extract(x)
        x = self.first_residual(x)
        x = self.second_residual(x)
        x = self.third_residual(x)
        x = self.fourth_residual(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        if x.dim() == 5:
            batch_size, num_crops, C, H, W = x.shape
            x = x.view(batch_size * num_crops, C, H, W)
            outputs = self.forward_single_crop(x)
            outputs = outputs.view(batch_size, num_crops, -1)
            if self.use_simple_fusion:
                x = outputs.mean(dim=1)
            else:
                outputs_flat = outputs.view(batch_size, -1)
                x = self.crop_fusion(outputs_flat)
        else:
            x = self.forward_single_crop(x)
        return x


if __name__ == "__main__":
    model = CNNmodel(num_classes=7, use_simple_fusion=True)
    x = torch.randn(4, 5, 1, 64, 64)
    output = model(x)
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x_single = torch.randn(4, 1, 64, 64)
    output_single = model(x_single)
    print(f"Single crop output shape: {output_single.shape}")
