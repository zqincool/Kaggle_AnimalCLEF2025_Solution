import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super(DCGANGenerator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            # State: N x (feature_g*8) x 4 x 4
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # State: N x (feature_g*4) x 8 x 8
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # State: N x (feature_g*2) x 16 x 16
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # State: N x feature_g x 32 x 32
            nn.ConvTranspose2d(feature_g, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()
            # Output: N x img_channels x 32 x 32
        )

    def forward(self, x):
        return self.net(x)

class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(DCGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x img_channels x 32 x 32
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: N x feature_d x 16 x 16
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: N x (feature_d*2) x 8 x 8
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: N x (feature_d*4) x 4 x 4
            nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: N x 1 x 1 x 1
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1) 