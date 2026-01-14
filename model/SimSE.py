class SimSE(nn.Module):
    def __init__(self, channels, reduction=16, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def get_simam_attention(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        e_inv = x_minus_mu_square / (4 * (y + self.e_lambda)) + 0.5
        return torch.sigmoid(e_inv)

    def forward(self, x):

        spatial_att = self.get_simam_attention(x)
        x_refined = x * spatial_att
        
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x_refined * y