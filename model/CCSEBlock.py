import torch
import torch.nn as nn

class CrossChannelSegmentationExcitationBlock(nn.Module): 
    def __init__(self, channels, reduction=16): 
        super().__init__() 
        self.half = channels // 2 
        mid_channels = max(self.half // reduction, 1) 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        
        self.mlp_a = nn.Sequential( 
            nn.Linear(channels - self.half, mid_channels, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(mid_channels, self.half, bias=False), 
            nn.Sigmoid() 
        ) 
        self.mlp_b = nn.Sequential( 
            nn.Linear(self.half, mid_channels, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(mid_channels, channels - self.half, bias=False), 
            nn.Sigmoid() 
        ) 

    def forward(self, x): 
        b, c, h, w = x.shape 
        
        x_a = x[:, :self.half, :, :] 
        x_b = x[:, self.half:, :, :] 

        z_a = self.avg_pool(x_a).view(b, self.half) 
        z_b = self.avg_pool(x_b).view(b, c - self.half) 
        
        w_a = self.mlp_a(z_b).view(b, self.half, 1, 1) 
        w_b = self.mlp_b(z_a).view(b, c - self.half, 1, 1) 
        
        out_a = x_a * w_a 
        out_b = x_b * w_b 
        
        out = torch.cat([out_a, out_b], dim=1)
        
        b, c, h, w = out.shape
        out = out.view(b, 2, c // 2, h, w)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, c, h, w)
        return out
