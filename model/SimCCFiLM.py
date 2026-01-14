import torch
import torch.nn as nn

class SimCCFiLM(nn.Module):
    def __init__(self, channels, reduction=16, alpha=0.45, beta_scale=0.05, e_lambda=1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta_scale = beta_scale
        self.e_lambda = e_lambda  
        
        mid_channels = max(channels // reduction, 1)

        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels * 2, bias=False)
        )
        
        nn.init.zeros_(self.fc[-1].weight)

    def get_simam_attention(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        
        e_inv = x_minus_mu_square / (4 * (y + self.e_lambda)) + 0.5

        return torch.sigmoid(e_inv)

    def forward(self, x):
        b, c, h, w = x.shape
        
        spatial_att = self.get_simam_attention(x)
        x_refined = x * spatial_att

        z_avg = self.pool_avg(x).view(b, c)
        z_max = self.pool_max(x).view(b, c)
        z = z_avg + z_max

        params = self.fc(z).view(b, 2, c, 1, 1)
        g, b_val = params[:, 0], params[:, 1]

        g = 1.0 + self.alpha * torch.tanh(g)
        b_val = self.beta_scale * torch.tanh(b_val)

        return x_refined * g + b_val