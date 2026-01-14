import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from SEBlock import SELayer
from ECABlock import ECALayer
from CCFiLM import CCFiLM
from SimCCFiLM import SimCCFiLM
from CoordAttBlock import CoordAtt
from SimAM import SimAM
from EMA import EMA


class AttentionLayer(nn.Module):
    """Generic attention layer that can use different attention mechanisms"""
    def __init__(self, channel, attention_type='se', **kwargs):
        super(AttentionLayer, self).__init__()
        
        self.attention_type = attention_type.lower()
        
        if self.attention_type == 'se':
            reduction = kwargs.get('reduction', 16)
            self.attention = SELayer(channel, reduction)
        elif self.attention_type == 'eca':
            self.attention = ECALayer(channel)
        elif self.attention_type == 'ccfilm':
            reduction = kwargs.get('reduction', 16)
            self.attention = CCFiLM(channel, reduction)
        elif self.attention_type == 'simccfilm':
            reduction = kwargs.get('reduction', 16)
            self.attention = SimCCFiLM(channel, reduction)
        elif self.attention_type == 'coordatt':
            reduction = kwargs.get('reduction', 32)
            self.attention = CoordAtt(channel, channel, reduction)
        elif self.attention_type == 'simam':
            e_lambda = kwargs.get('e_lambda', 1e-4)
            self.attention = SimAM(e_lambda=e_lambda)
        elif self.attention_type == 'ema':
            factor = kwargs.get('factor', 32)
            self.attention = EMA(channel, factor=factor)
        elif self.attention_type == 'none':
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
    
    def forward(self, x):
        return self.attention(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='none', **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.attention = AttentionLayer(planes, attention_type, **kwargs)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='none', **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Fix: Use correct channel count for attention layer (planes * expansion)
        self.attention = AttentionLayer(planes * 4, attention_type, **kwargs)

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

        # Apply attention mechanism after conv3 and batch norm, before residual addition
        out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, attention_type='none', input_channels=3, pretrained=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.attention_type = attention_type.lower()
        self.pretrained = pretrained
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], attention_type=attention_type, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, attention_type=attention_type, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention_type=attention_type, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention_type=attention_type, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights or load pretrained
        if pretrained:
            self._load_pretrained_weights(block, layers, num_classes, input_channels)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self, block, layers, num_classes, input_channels):
        """Load pretrained weights from torchvision models, adapting for attention mechanisms"""
        # Map our block types to standard ResNet blocks for loading
        if block == BasicBlock:
            if layers == [2, 2, 2, 2]:
                # ResNet-18
                try:
                    # Try to use the newer weights parameter
                    pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                except AttributeError:
                    # Fallback to deprecated pretrained parameter
                    pretrained_model = models.resnet18(pretrained=True)
            elif layers == [3, 4, 6, 3]:
                # ResNet-34
                try:
                    pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                except AttributeError:
                    pretrained_model = models.resnet34(pretrained=True)
            else:
                # For custom configurations, initialize randomly
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                return
        elif block == Bottleneck:
            if layers == [3, 4, 6, 3]:
                # ResNet-50
                try:
                    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                except AttributeError:
                    pretrained_model = models.resnet50(pretrained=True)
            elif layers == [3, 4, 23, 3]:
                # ResNet-101
                try:
                    pretrained_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                except AttributeError:
                    pretrained_model = models.resnet101(pretrained=True)
            else:
                # For custom configurations, initialize randomly
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                return
        else:
            # For custom configurations, initialize randomly
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            return
        
        # Load state dict with modifications for attention layers
        self._transfer_weights(pretrained_model.state_dict(), input_channels, num_classes)
    
    def _transfer_weights(self, pretrained_state_dict, input_channels, num_classes):
        """Transfer weights from pretrained model, adapting for attention layers and different input/output shapes"""
        own_state = self.state_dict()
        transferred_layers = set()
        
        for name, param in pretrained_state_dict.items():
            # Skip layers that we don't want to transfer
            if name.startswith(('fc.', 'layer4.2.attention.', 'layer3.5.attention.')):  # Skip attention layers and final FC
                continue
                
            # Handle different input channel counts and kernel sizes
            if name == 'conv1.weight':
                # Original torchvision ResNet uses 7x7 kernel, but ours uses 3x3 kernel
                # Extract center weights if kernel sizes differ
                if input_channels == 3 and param.size(1) == 3:
                    # Standard RGB input
                    if own_state[name].shape == param.shape:
                        # Same kernel size, direct copy
                        own_state[name].copy_(param)
                        transferred_layers.add(name)
                    else:
                        # Different kernel sizes, extract center portion
                        if param.shape[2] == 7 and own_state[name].shape[2] == 3:  # Pretrained 7x7 -> Our 3x3
                            center_start = (7 - 3) // 2
                            center_end = center_start + 3
                            own_state[name].copy_(param[:, :, center_start:center_end, center_start:center_end])
                            transferred_layers.add(name)
                        elif param.shape[2] == 3 and own_state[name].shape[2] == 7:  # Our 7x7 <- Pretrained 3x3
                            center_start = (7 - 3) // 2
                            center_end = center_start + 3
                            own_state[name][:, :, center_start:center_end, center_start:center_end].copy_(param)
                            transferred_layers.add(name)
                        else:
                            # Other size combinations - initialize randomly
                            continue
                elif input_channels == 1 and param.size(1) == 3:
                    # Convert RGB to grayscale by averaging, then handle kernel size
                    gray_param = param.sum(dim=1, keepdim=True) / 3.0
                    if own_state[name].shape == gray_param.shape:
                        own_state[name].copy_(gray_param)
                        transferred_layers.add(name)
                    else:
                        # Different kernel sizes for grayscale
                        if gray_param.shape[2] == 7 and own_state[name].shape[2] == 3:  # Pretrained 7x7 -> Our 3x3
                            center_start = (7 - 3) // 2
                            center_end = center_start + 3
                            own_state[name].copy_(gray_param[:, :, center_start:center_end, center_start:center_end])
                            transferred_layers.add(name)
                        elif gray_param.shape[2] == 3 and own_state[name].shape[2] == 7:  # Our 7x7 <- Pretrained 3x3
                            center_start = (7 - 3) // 2
                            center_end = center_start + 3
                            own_state[name][:, :, center_start:center_end, center_start:center_end].copy_(gray_param)
                            transferred_layers.add(name)
                elif input_channels == 3 and param.size(1) == 1:
                    # Convert grayscale to RGB by repeating, then handle kernel size
                    rgb_param = param.repeat(1, 3, 1, 1) / 3.0
                    if own_state[name].shape == rgb_param.shape:
                        own_state[name].copy_(rgb_param)
                        transferred_layers.add(name)
                    else:
                        # Different kernel sizes for RGB
                        if rgb_param.shape[2] == 7 and own_state[name].shape[2] == 3:  # Pretrained 7x7 -> Our 3x3
                            center_start = (7 - 3) // 2
                            center_end = center_start + 3
                            own_state[name].copy_(rgb_param[:, :, center_start:center_end, center_start:center_end])
                            transferred_layers.add(name)
                        elif rgb_param.shape[2] == 3 and own_state[name].shape[2] == 7:  # Our 7x7 <- Pretrained 3x3
                            center_start = (7 - 3) // 2
                            center_end = center_start + 3
                            own_state[name][:, :, center_start:center_end, center_start:center_end].copy_(rgb_param)
                            transferred_layers.add(name)
                else:
                    # Different channel count - initialize randomly
                    continue
            elif name == 'fc.weight':
                if own_state[name].size() == param.size():
                    # Only copy if output classes match and tensor shapes match
                    own_state[name].copy_(param)
                    transferred_layers.add(name)
            elif name == 'fc.bias':
                if own_state[name].size() == param.size():
                    # Only copy if output classes match and tensor shapes match
                    own_state[name].copy_(param)
                    transferred_layers.add(name)
            else:
                # Copy other layers directly if they exist in our model and shapes match
                if name in own_state and own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                    transferred_layers.add(name)
        
        print(f"Transferred {len(transferred_layers)} layers from pretrained model")

    def _make_layer(self, block, planes, blocks, stride=1, attention_type='none', **kwargs):            
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention_type, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention_type=attention_type, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
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
        
        return x


def resnet18(num_classes=1000, input_channels=3, attention_type='none', pretrained=False, **kwargs):
    """Constructs a ResNet-18 model with specified attention mechanism."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                  input_channels=input_channels, attention_type=attention_type, pretrained=pretrained, **kwargs)

def resnet34(num_classes=1000, input_channels=3, attention_type='none', pretrained=False, **kwargs):
    """Constructs a ResNet-34 model with specified attention mechanism."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, 
                  input_channels=input_channels, attention_type=attention_type, pretrained=pretrained, **kwargs)

def resnet50(num_classes=1000, input_channels=3, attention_type='none', pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with specified attention mechanism."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, 
                  input_channels=input_channels, attention_type=attention_type, pretrained=pretrained, **kwargs)

def resnet101(num_classes=1000, input_channels=3, attention_type='none', pretrained=False, **kwargs):
    """Constructs a ResNet-101 model with specified attention mechanism."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, 
                  input_channels=input_channels, attention_type=attention_type, pretrained=pretrained, **kwargs)


# Specific attention mechanism models
def se_resnet18(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """SE-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='se', reduction=reduction, pretrained=pretrained)

def se_resnet34(num_classes=1000, input_channels=3, reduction=16, pretrained=False):    
    """SE-ResNet-34 model"""
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='se', reduction=reduction, pretrained=pretrained)

def se_resnet50(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """SE-ResNet-50 model"""
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='se', reduction=reduction, pretrained=pretrained)

def se_resnet101(num_classes=1000, input_channels=3, reduction=16, pretrained=False):   
    """SE-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='se', reduction=reduction, pretrained=pretrained)


def eca_resnet18(num_classes=1000, input_channels=3, pretrained=False):
    """ECA-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='eca', pretrained=pretrained )

def eca_resnet34(num_classes=1000, input_channels=3, pretrained=False):
    """ECA-ResNet-34 model"""
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='eca', pretrained=pretrained)

def eca_resnet50(num_classes=1000, input_channels=3, pretrained=False):
    """ECA-ResNet-50 model"""
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='eca', pretrained=pretrained)

def eca_resnet101(num_classes=1000, input_channels=3, pretrained=False):
    """ECA-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='eca', pretrained=pretrained)


def ccfilm_resnet18(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """CCFiLM-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ccfilm', reduction=reduction, pretrained=pretrained)

def ccfilm_resnet34(num_classes=1000, input_channels=3, reduction=16, pretrained=False):  
    """CCFiLM-ResNet-34 model"""    
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ccfilm', reduction=reduction, pretrained=pretrained)

def ccfilm_resnet50(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """CCFiLM-ResNet-50 model"""    
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ccfilm', reduction=reduction, pretrained=pretrained)

def ccfilm_resnet101(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """CCFiLM-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ccfilm', reduction=reduction, pretrained=pretrained)

def simccfilm_resnet18(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """SimCCFiLM-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simccfilm', reduction=reduction, pretrained=pretrained)

def simccfilm_resnet34(num_classes=1000, input_channels=3, reduction=16, pretrained=False):  
    """SimCCFiLM-ResNet-34 model"""
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simccfilm', reduction=reduction, pretrained=pretrained)

def simccfilm_resnet50(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """SimCCFiLM-ResNet-50 model"""
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simccfilm', reduction=reduction, pretrained=pretrained)

def simccfilm_resnet101(num_classes=1000, input_channels=3, reduction=16, pretrained=False):
    """SimCCFiLM-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simccfilm', reduction=reduction, pretrained=pretrained)

def coordatt_resnet18(num_classes=1000, input_channels=3, reduction=32, pretrained=False):  
    """CoordAtt-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='coordatt', reduction=reduction, pretrained=pretrained)

def coordatt_resnet34(num_classes=1000, input_channels=3, reduction=32, pretrained=False):      
    """CoordAtt-ResNet-34 model"""
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='coordatt', reduction=reduction, pretrained=pretrained)

def coordatt_resnet50(num_classes=1000, input_channels=3, reduction=32, pretrained=False):
    """CoordAtt-ResNet-50 model"""
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='coordatt', reduction=reduction, pretrained=pretrained)

def coordatt_resnet101(num_classes=1000, input_channels=3, reduction=32, pretrained=False):
    """CoordAtt-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='coordatt', reduction=reduction, pretrained=pretrained)


def simam_resnet18(num_classes=1000, input_channels=3, e_lambda=1e-4, pretrained=False):
    """SimAM-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simam', e_lambda=e_lambda, pretrained=pretrained)

def simam_resnet34(num_classes=1000, input_channels=3, e_lambda=1e-4, pretrained=False):
    """SimAM-ResNet-34 model"""
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simam', e_lambda=e_lambda, pretrained=pretrained)

def simam_resnet50(num_classes=1000, input_channels=3, e_lambda=1e-4, pretrained=False):
    """SimAM-ResNet-50 model"""
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simam', e_lambda=e_lambda, pretrained=pretrained)

def simam_resnet101(num_classes=1000, input_channels=3, e_lambda=1e-4, pretrained=False):
    """SimAM-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='simam', e_lambda=e_lambda, pretrained=pretrained)


def ema_resnet18(num_classes=1000, input_channels=3, factor=32, pretrained=False):
    """EMA-ResNet-18 model"""
    return resnet18(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ema', factor=factor, pretrained=pretrained)

def ema_resnet34(num_classes=1000, input_channels=3, factor=32, pretrained=False):
    """EMA-ResNet-34 model"""
    return resnet34(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ema', factor=factor, pretrained=pretrained)

def ema_resnet50(num_classes=1000, input_channels=3, factor=32, pretrained=False):
    """EMA-ResNet-50 model"""
    return resnet50(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ema', factor=factor, pretrained=pretrained)

def ema_resnet101(num_classes=1000, input_channels=3, factor=32, pretrained=False):
    """EMA-ResNet-101 model"""
    return resnet101(num_classes=num_classes, input_channels=input_channels, 
                    attention_type='ema', factor=factor, pretrained=pretrained)


if __name__ == "__main__":
    # Test different attention mechanisms
    attention_types = ['se', 'eca', 'ccfilm', 'simccfilm', 'coordatt', 'simam', 'ema', 'none']
    
    for attn_type in attention_types:
        print(f"\nTesting {attn_type.upper()}-ResNet-18:")
        try:
            if attn_type == 'se':
                model = se_resnet18(num_classes=7, input_channels=1, reduction=16)
            elif attn_type == 'eca':
                model = eca_resnet18(num_classes=7, input_channels=1)
            elif attn_type == 'ccfilm':
                model = ccfilm_resnet18(num_classes=7, input_channels=1, reduction=16)
            elif attn_type == 'coordatt':
                model = coordatt_resnet18(num_classes=7, input_channels=1, reduction=32)
            elif attn_type == 'simam':
                model = simam_resnet18(num_classes=7, input_channels=1, e_lambda=1e-4)
            elif attn_type == 'ema':
                model = ema_resnet18(num_classes=7, input_channels=1, factor=32)
            else:
                model = resnet18(num_classes=7, input_channels=1, attention_type=attn_type)
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameter count: {total_params / 1e6:.6f}M")

            input_tensor = torch.randn(1, 1, 48, 48)
            output = model(input_tensor)
            print(f"  Input shape: {input_tensor.shape}")
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"  Error with {attn_type}: {e}")