import torch
import torch.nn as nn

class OpenGANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super(OpenGANGenerator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, feature_g * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # State: N x (feature_g*4) x 4 x 4
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # State: N x (feature_g*2) x 8 x 8
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # State: N x feature_g x 16 x 16
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: N x img_channels x 32 x 32
        )

    def forward(self, x):
        return self.net(x) 