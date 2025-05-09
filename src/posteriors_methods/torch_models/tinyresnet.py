"""Implements the bayesmates TinyResNet in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock(nn.Module):
    """Basic ResNet block with two convolutional layers."""
    def __init__(self, in_channels, out_channels, stride=1, activation=F.relu):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Downsample if the input and output dimensions differ or if stride is not 1.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.activation(out)
        return out

class TinyResNetCore(nn.Module):
    """Core implementation of a tiny ResNet."""
    def __init__(self, num_classes, activation=F.relu):
        super(TinyResNetCore, self).__init__()
        self.activation = activation
        # Assuming input image has 3 channels (RGB)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block = BasicBlock(64, 64, stride=1, activation=activation)
        # Global average pooling over spatial dimensions.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward_features(self, x):
        """Extract features from the model before the final fully connected layer."""
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # Return features before the final fully connected layer
    
    def forward_classifier(self, x):
        """Apply the final fully connected layer to the features."""
        x = self.fc(x)
        return x

# A simple config class for TinyResNet.
class TinyResNetConfigTorch:
    def __init__(self, out_dim, activation=F.relu):
        self.out_dim = out_dim
        self.activation = activation

class TinyResNet(nn.Module):
    """A tiny ResNet model."""
    def __init__(self, config: TinyResNetConfigTorch):
        super(TinyResNet, self).__init__()
        self.core = TinyResNetCore(num_classes=config.out_dim, activation=config.activation)

    def forward(self, x):
        return self.core(x)