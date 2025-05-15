import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(SimpleCNN, self).__init__()
        self.layer1 = BasicBlock(in_channels, 32)
        self.layer2 = BasicBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer3 = BasicBlock(64, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # for input size 32x32
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
