import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=nn.ReLU, bias=False, bn=True):
        super(ConvBlock, self).__init__()
        padding = (kernel_size) // 2
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        # Batch normalization (if enabled)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        # Activation function
        self.activation = activation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.
        """
        return self.activation(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        # Squeeze-and-Excitation reduction ratio
        r = in_channels // 4
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers for channel-wise attention
        self.fc1 = nn.Linear(in_channels, r)
        self.fc2 = nn.Linear(r, in_channels)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.globalpooling(x)
        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.hsigmoid(self.fc2(out))
        out = out[:, :, None, None]  # Reshape to [B, C, 1, 1]
        
        # Apply channel-wise attention
        scale = x * out
        return scale

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, se, activation):
        super(BottleNeckBlock, self).__init__()
        # Check if we need to use residual connection
        self.add = in_channels == out_channels and stride == 1
        # Inverted residual structure
        self.bnblock = nn.Sequential(
            ConvBlock(in_channels, expansion, 1, 1, activation=activation),  # Expansion
            ConvBlock(expansion, expansion, kernel_size, stride, activation=activation, groups=expansion),  # Depthwise
            SEBlock(expansion) if se else nn.Identity(),  # Squeeze-and-Excitation (if enabled)
            ConvBlock(expansion, out_channels, 1, 1, activation=nn.Identity)  # Projection
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bnblock(x)
        if self.add:
            out += x  # Residual connection
        return out

class MobileNetv3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNetv3, self).__init__()
        # Initial convolution layer
        self.conv1 = ConvBlock(in_channels, 16, 3, 2, activation=nn.Hardswish)    
        
        # Building inverted residual blocks
        self.blocks = nn.ModuleList([])
        bneck_settings = [
            # kernel, expansion, in_channels, out_channels, SEBlock, activation, stride
            # ... (settings for each block)
        ]
        
        for setting in bneck_settings:
            kernel, expansion, in_channels, out_channels, se, activation, stride = setting
            self.blocks.append(BottleNeckBlock(in_channels, out_channels, kernel, stride, expansion, se, activation))
            
        # Final layers and classifier
        last_out_channels = 160
        last_expansion = 960
        self.classifier = nn.Sequential(
            ConvBlock(last_out_channels, last_expansion, 1, 1, activation=nn.Hardswish),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            ConvBlock(last_expansion, 1280, 1, 1, activation=nn.Hardswish, bn=False, bias=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)  # Final classification layer
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        for block in self.blocks:
            out = block(out)
        out = self.classifier(out)
        return torch.flatten(out, 1)  # Flatten the output for classification