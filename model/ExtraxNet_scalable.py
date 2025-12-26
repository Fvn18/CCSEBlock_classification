import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .CCSEBlock import CrossChannelSegmentationExcitationBlock
except ImportError:
    from CCSEBlock import CrossChannelSegmentationExcitationBlock



class LiteResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, expand_ratio=3, use_ccse=True):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_c * expand_ratio))
        self.use_res_connect = (self.stride == 1 and in_c == out_c)
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        else:
            hidden_dim = in_c
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        if use_ccse:
            layers.append(CrossChannelSegmentationExcitationBlock(hidden_dim, reduction=8))
        layers.extend([
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ExtraNet_Scalable(nn.Module):
    def __init__(self, config, num_classes=7, input_channels=3):
        super(ExtraNet_Scalable, self).__init__()
        stem_width = config['stem_width']
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, max(1, stem_width // 2), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(max(1, stem_width // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, stem_width // 2), stem_width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        input_channel = stem_width
        for stage_idx, (out_c, num_blocks, stride, expand_ratio) in enumerate(config['stages']):
            stage_layers = []
            stage_layers.append(LiteResidualBlock(
                input_channel, out_c, stride=stride, expand_ratio=expand_ratio
            ))
            input_channel = out_c
            for _ in range(1, num_blocks):
                stage_layers.append(LiteResidualBlock(
                    input_channel, out_c, stride=1, expand_ratio=expand_ratio
                ))
            self.blocks.append(nn.Sequential(*stage_layers))
        
        last_channel = input_channel
        head_dim = config.get('head_dim', last_channel)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(last_channel, head_dim, 1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(head_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_single(self, x):
        x = self.stem(x)
        for stage in self.blocks:
            x = stage(x)
        x = self.final_conv(x)
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

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def get_extranet_config(scale='tiny'):
    base_config = {
        'stem_width': 32,
        'stages': [
            [48,  4, 2, 3],
            [80,  5, 2, 4],
            [144, 10, 2, 4],
            [256, 4, 1, 4],
        ],
        'head_dim': 960,
        'dropout': 0.25
    }

    scale_factors = {
        'pico':   (0.35, 0.3, 0.1),
        'nano':   (0.5,  0.4, 0.15),
        'tiny':   (0.75, 0.6, 0.2),
        'base':   (1.0,  1.0, 0.25),
        'large':  (1.25, 1.5, 0.3),
        'xlarge': (1.5,  2.0, 0.35),
        'huge':   (2.0,  2.8, 0.4),
    }

    if scale not in scale_factors:
        raise ValueError(f"Invalid scale: {scale}. Available scales: {list(scale_factors.keys())}")

    w_mult, d_mult, dropout = scale_factors[scale]

    new_config = {}
    
    new_config['stem_width'] = _make_divisible(base_config['stem_width'] * w_mult)
    new_config['head_dim'] = _make_divisible(base_config['head_dim'] * w_mult)
    new_config['dropout'] = dropout

    new_stages = []
    for out_c, num_blocks, stride, expand_ratio in base_config['stages']:
        scaled_out_c = _make_divisible(out_c * w_mult)
        scaled_num_blocks = max(1, int(round(num_blocks * d_mult)))
        new_stages.append([scaled_out_c, scaled_num_blocks, stride, expand_ratio])
    
    new_config['stages'] = new_stages
    
    return new_config

def extra_net(scale='tiny', num_classes=7, input_channels=1):
    config = get_extranet_config(scale)
    model = ExtraNet_Scalable(config, num_classes=num_classes, input_channels=input_channels)
    return model

if __name__ == "__main__":
    scales = ['pico', 'nano', 'tiny', 'base', 'large', 'xlarge', 'huge']
    
    resolutions = {
        'pico': 64, 'nano': 80, 'tiny': 96, 'base': 112, 
        'large': 128, 'xlarge': 144, 'huge': 160
    }

    print(f"{'Scale':<10} | {'Params (M)':<12} | {'Size (MB)':<11} | {'Input Res':<11} | {'Output Shape'}")
    print("-" * 70)
    
    for s in scales:
        res = resolutions.get(s, 96)
        dummy_input_tta = torch.randn(2, 14, 1, res, res) 
        dummy_input_train = torch.randn(4, 1, res, res)
        
        model = extra_net(s)
        model.eval()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = params * 4 / (1024 * 1024)
        
        with torch.no_grad():
            out_tta = model(dummy_input_tta)
            out_train = model(dummy_input_train)
            
        print(f"{s.capitalize():<10} | {params/1e6:<12.2f} | {model_size_mb:<11.2f} | {res:<11} | {str(list(out_train.shape)):<15}")
    
    print("\nVerifying TTA and training path outputs are consistent:")
    print(f"Train path output shape: {out_train.shape}")
    print(f"TTA path output shape:   {out_tta.shape}")
