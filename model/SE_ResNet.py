import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ifUseSE=True):
        super(SEBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if ifUseSE:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ifUseSE=True):
        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        if ifUseSE:
            self.se = SELayer(planes * 4, reduction)
        else:
            self.se = nn.Identity()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, reduction=16, input_channels=3, ifUseSE=True):
        super(SEResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction, ifUseSE=ifUseSE)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction, ifUseSE=ifUseSE)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction, ifUseSE=ifUseSE)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction, ifUseSE=ifUseSE)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16, ifUseSE=True):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction, ifUseSE))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction, ifUseSE=ifUseSE))

        return nn.Sequential(*layers)

    def forward(self, x):

        input_shape = x.shape
        is_tencrop = len(input_shape) == 5  
        
        if is_tencrop:
            batch_size, num_crops = input_shape[0], input_shape[1]
            x = x.view(batch_size * num_crops, *input_shape[2:])
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if is_tencrop:
            x = x.view(batch_size, num_crops, -1)
            x = x.mean(dim=1)
        
        return x



def se_resnet18(num_classes=1000, input_channels=3):
    """Constructs a SE-ResNet-18 model."""
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels, ifUseSE=True)

def se_resnet34(num_classes=1000, input_channels=3):
    """Constructs a SE-ResNet-34 model."""
    return SEResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=True)   
def se_resnet50(num_classes=1000, input_channels=3):
    """Constructs a SE-ResNet-50 model."""
    return SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=True)

def se_resnet101(num_classes=1000, input_channels=3):
    """Constructs a SE-ResNet-101 model."""
    return SEResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=True)
def se_resnet152(num_classes=1000, input_channels=3):
    """Constructs a SE-ResNet-152 model."""
    return SEResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=True)

def resnet18(num_classes=1000, input_channels=3):
    """Constructs a ResNet-18 model without SE blocks."""
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels, ifUseSE=False)

def resnet34(num_classes=1000, input_channels=3):
    """Constructs a ResNet-34 model without SE blocks."""
    return SEResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=False)

def resnet50(num_classes=1000, input_channels=3):
    """Constructs a ResNet-50 model without SE blocks."""
    return SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=False)

def resnet101(num_classes=1000, input_channels=3):
    """Constructs a ResNet-101 model without SE blocks."""
    return SEResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=False)

def resnet152(num_classes=1000, input_channels=3):
    """Constructs a ResNet-152 model without SE blocks."""
    return SEResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, input_channels=input_channels, ifUseSE=False) 


if __name__ == "__main__":

    model = resnet18(num_classes=7, input_channels=1)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SE-ResNet-50 parameter count: {total_params / 1e6:.6f}M")

    input_tensor = torch.randn(1, 1, 48, 48)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}") 
