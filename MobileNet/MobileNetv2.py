import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expansion_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        
        # Calculate the expanded dimension
        expansion_dim = input_channels * expansion_ratio
        # Determine if we should use a residual connection
        self.residual_connection = self.stride == 1 and input_channels == output_channels
        
        layers = []
        # If expansion ratio is not 1, add pointwise convolution to expand channels
        if expansion_ratio != 1:
            layers.append(nn.Conv2d(input_channels, expansion_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expansion_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Add depthwise convolution, followed by pointwise convolution to reduce channels
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(expansion_dim, expansion_dim, kernel_size=3, stride=stride, padding=1, groups=expansion_dim, bias=False),
            nn.BatchNorm2d(expansion_dim),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(expansion_dim, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels)
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # Apply residual connection if applicable
        if self.residual_connection:
            out = x + self.layers(x)
        else:
            out = self.layers(x)
        return out

class MobileNetv2(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super(MobileNetv2, self).__init__()
        input_channel = 32
        # Configuration for each inverted residual block
        # [expansion ratio, output channels, number of repeats, stride]
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(input_channel * width_multiplier)
        
        # Initial convolution layer
        self.layers = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.layers.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        # Final pointwise convolution
        self.last_channel = int(1280 * max(1.0, width_multiplier))
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True),
        ))
        
        self.features = nn.Sequential(*self.layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # Apply feature extraction
        x = self.features(x)
        # Global average pooling
        x = x.mean([2, 3])
        # Apply classifier
        x = self.classifier(x)
        return x