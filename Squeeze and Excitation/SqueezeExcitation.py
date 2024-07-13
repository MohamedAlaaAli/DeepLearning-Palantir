import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.data_loader import load_cifar10
from Utils.trainer import train_and_validate
from ResNet import ResBlock, ResNet


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation module for the ResNet architecture.
    """
    def __init__(self, C, r=16):
        super(SEBlock, self).__init__()
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=C, out_features=C//r)
        self.fc2 = nn.Linear(in_features=C//r, out_features=C)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # shape of x is N, C, H,W
        f = self.glob_avg_pool(x)
        f = torch.flatten(f, 1)
        f = self.fc1(f)
        f = self.relu(f)
        f = self.fc2(f)
        f = self.sigmoid(f) # shape is N,C
        f = f[:,:, None, None] # shape is N,C,1,1

        return x*f


class Residual_SE(ResBlock):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual_SE, self).__init__(in_channels, out_channels, stride)
        self.se_block = SEBlock(out_channels)
    

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se_block(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def ResNet_SE(num_classes=10):
    return ResNet(Residual_SE, [3, 4, 6, 3], num_classes)

train_loader, val_loader = load_cifar10(8)


# Instantiate the model
model = ResNet_SE(num_classes=10)
model = model.to("cuda")

# Train and validate the model
train_and_validate(model, train_loader, val_loader, epochs=10, lr=0.001)
