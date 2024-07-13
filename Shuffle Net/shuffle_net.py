import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.data_loader import load_cifar10
from Utils.trainer import trainer

class Shuffle(nn.Module):
    """
    Shuffle layer to rearrange the channels of a tensor for the ShuffleNet architecture.
    """

    def __init__(self, groups):
        """
        Initializes the Shuffle layer.

        Args:
            groups (int): Number of groups for shuffling.
        """
        super().__init__()
        self.groups = groups

    def forward(self, x):
        """
        Forward pass of the Shuffle layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Shuffled tensor of shape (N, C, H, W).
        """
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):
    """
    Bottleneck layer used in the ShuffleNet architecture.
    """

    def __init__(self, input_channel, output_channel, stride, groups):
        """
        Initializes the Bottleneck layer.

        Args:
            input_channel (int): Number of input channels.
            output_channel (int): Number of output channels.
            stride (int): Stride for the convolutional layer.
            groups (int): Number of groups for group convolution.
        """
        super().__init__()
        self.stride = stride

        in_between_channel = int(output_channel / 4)
        g = 1 if input_channel == 24 else groups

        # Group Convolution
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(input_channel, in_between_channel, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(in_between_channel), 
            nn.ReLU(inplace=True)
        )
        self.shuffle = Shuffle(groups=g)

        # Depthwise Convolution
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_between_channel, in_between_channel, kernel_size=3, stride=stride, padding=1, groups=in_between_channel, bias=False),
            nn.BatchNorm2d(in_between_channel), 
            nn.ReLU(inplace=True)
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(in_between_channel, output_channel, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(output_channel)
        )
            
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        """
        Forward pass of the Bottleneck layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        out = self.conv1x1_1(x)
        out = self.shuffle(out)
        out = self.conv1x1_2(out)
        out = self.conv1x1_3(out)
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class ShuffleNet(nn.Module):
    """
    ShuffleNet model for image classification.
    """

    def __init__(self, cfg, input_channel, n_classes):
        """
        Initializes the ShuffleNet model.

        Args:
            cfg (dict): Configuration dictionary with keys 'out', 'n_blocks', and 'groups'.
            input_channel (int): Number of input channels (3 for CIFAR-10).
            n_classes (int): Number of output classes (10 for CIFAR-10).
        """
        super().__init__()
        output_channels = cfg['out']
        n_blocks = cfg['n_blocks']
        groups = cfg['groups']
        self.in_channels = 24
        
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 24, kernel_size=3, stride=1, padding=1, bias=False),  # Modified for CIFAR-10
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # ShuffleNet layers
        self.layer1 = self.make_layer(output_channels[0], n_blocks[0], groups)
        self.layer2 = self.make_layer(output_channels[1], n_blocks[1], groups)
        self.layer3 = self.make_layer(output_channels[2], n_blocks[2], groups)
        
        # Classification layer
        self.linear = nn.Linear(output_channels[2], n_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def make_layer(self, out_channel, n_blocks, groups):
        """
        Creates a ShuffleNet layer.

        Args:
            out_channel (int): Number of output channels.
            n_blocks (int): Number of bottleneck blocks.
            groups (int): Number of groups for group convolution.

        Returns:
            nn.Sequential: A ShuffleNet layer.
        """
        layers = []
        for i in range(n_blocks):
            stride = 2 if i == 0 else 1
            cat_channels = self.in_channels if i == 0 else 0
            layers.append(Bottleneck(self.in_channels, out_channel - cat_channels, stride=stride, groups=groups))        
            self.in_channels = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ShuffleNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, n_classes).
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Configuration for ShuffleNet
cfg = {
    'out': [240, 480, 960],  # Output channels for each stage
    'n_blocks': [4, 8, 4],   # Number of bottleneck blocks for each stage
    'groups': 3              # Number of groups for group convolution
}

# Instantiate the model
model = ShuffleNet(cfg, input_channel=3, n_classes=10)
model = model.to("cuda")
train_loader, val_loader = load_cifar10(8)


# Train and validate the model
train_and_validate(model, train_loader, val_loader, epochs=10, lr=0.001)
