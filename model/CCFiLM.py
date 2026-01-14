import torch
import torch.nn as nn
import torch.nn.functional as F

class CCFiLM(nn.Module):

    def __init__(self, channels, reduction=16, alpha=0.45, beta_scale=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta_scale = beta_scale
        
        mid_channels = max(channels // reduction, 1)

        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels * 2, bias=False)
        )
        
        nn.init.zeros_(self.fc[-1].weight)

    def forward(self, x):
        b, c, _, _ = x.shape
        
        z_avg = self.pool_avg(x).view(b, c)
        z_max = self.pool_max(x).view(b, c)
        z = z_avg + z_max
        params = self.fc(z).view(b, 2, c, 1, 1)
        gamma, beta = params[:, 0], params[:, 1]

        gamma = 1.0 + self.alpha * torch.tanh(gamma)
        beta = self.beta_scale * torch.tanh(beta)

        return x * gamma + beta

