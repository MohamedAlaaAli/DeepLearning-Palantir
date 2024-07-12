import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1(nn.Module):
    def __init__(self, in_channels=3, out_channels=24):
        super(Stage1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class GConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(GConv, self).__init__()
        self.groups = groups
        self.group_convs = nn.ModuleList([
            nn.Conv2d(in_channels // groups, out_channels // groups, kernel_size=1)
            for _ in range(groups)
        ])
    
    def forward(self, x):
        group_outs = torch.chunk(x, self.groups, dim=1)
        group_outs = [conv(g) for conv, g in zip(self.group_convs, group_outs)]
        return torch.cat(group_outs, dim=1)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

class ShuffleNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetBlock, self).__init__()
        self.stride = stride
        mid_channels = in_channels // 4
        self.gconv1 = GConv(in_channels, mid_channels, groups)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.channel_shuffle = ChannelShuffle(groups)
        self.dwsc = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.gconv2 = GConv(mid_channels, out_channels, groups)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride == 2:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        x = self.gconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.channel_shuffle(x)
        x = self.dwsc(x)
        x = self.bn2(x)
        x = self.gconv2(x)
        x = self.bn3(x)
        if self.stride == 2:
            residual = self.avgpool(residual)
            x = torch.cat((residual, x), 1)
        else:
            x = x + residual
        print(f'ShuffleNetBlock output shape: {x.shape}')
        return F.relu(x)

class ShuffleNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, groups):
        super(ShuffleNetStage, self).__init__()
        self.blocks = nn.ModuleList()

        # Handle the first block with stride 2 separately
        self.blocks.append(ShuffleNetBlock(in_channels, out_channels, stride=2, groups=groups))
        for i in range(1, num_blocks):  # Start from the second block (index 1)
            self.blocks.append(ShuffleNetBlock(out_channels, out_channels, stride=1, groups=groups))

    def forward(self, x):
        residual = x  # Initialize residual for the first block
        for block in self.blocks:
            out = block(x)
            x = out + residual  # Add residual for all blocks (including the first)
            residual = out  # Update residual for next block
        return x


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.stage1 = Stage1(in_channels=3, out_channels=24)
        self.stage2 = ShuffleNetStage(in_channels=24, out_channels=384, num_blocks=3, groups=8)
        self.stage3 = ShuffleNetStage(in_channels=384, out_channels=768, num_blocks=7, groups=8)
        self.stage4 = ShuffleNetStage(in_channels=768, out_channels=1536, num_blocks=3, groups=8)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        print(f'Final output shape: {x.shape}')
        return x

# Example usage
model = ShuffleNet()

# Define a sample input tensor with batch size 1 and image size 224x224 with 3 channels
sample_input = torch.randn(1, 3, 224, 224)

# Forward pass through the model
output = model(sample_input)
print("Output shape:", output.shape)
