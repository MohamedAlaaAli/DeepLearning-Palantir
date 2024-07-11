import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the basic building block of ResNet (a residual block)
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        #If the input and output dimensions do not match (either due to stride or channel number),
        #a 1x1 convolution and batch normalization are applied to the input to match the dimensions.
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet class using the basic blocks
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        """
        Initializes the ResNet model with the specified block type, number of blocks, and number of classes.

        Parameters:
            block (torch.nn.Module): The basic block type for the ResNet.
            num_blocks (list): A list of integers specifying the number of blocks in each layer.
            num_classes (int, optional): The number of output classes. Default is 1000.

        Initializes the convolutional layers, batch normalization layers, max pooling layer, residual layers, adaptive average pooling layer, and fully connected layer.
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        # define the stem blocks of the network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # define the residual layers of the network
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # define the average pooling layer and the fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Create a sequential layer by repeating a given block multiple times based on the number of blocks.
        
        Parameters:
            block: The block to be repeated.
            out_channels: The number of output channels in the block.
            num_blocks: The total number of blocks to create.
            stride: The stride value for each block.
        
        Returns:
            A sequential layer containing the repeated blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).

        This method performs the forward pass of the ResNet model. It takes an input tensor `x` and applies a series of convolutional layers, batch normalization layers, and activation functions. The input is first passed through a convolutional layer, batch normalization layer, and ReLU activation function. Then, it is passed through a max pooling layer. 

        The model then applies a series of residual blocks (`layer1`, `layer2`, `layer3`, `layer4`) which each consist of two convolutional layers, batch normalization layers, and activation functions. The residual connections allow the model to learn more complex and deeper representations of the input data.

        After the residual blocks, the input is passed through an average pooling layer and then flattened to a 1D tensor. Finally, the flattened tensor is passed through a fully connected layer (`fc`) to produce the output tensor.

        Note: The number of channels in the input tensor (`channels`) must match the number of channels in the first convolutional layer of the model.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet34(num_classes=1000):
    return ResNet(ResBlock, [3, 4, 6, 3], num_classes)

if __name__ == '__main__':
    # Example of creating a ResNet-34 model
    model = ResNet34(num_classes=1000)

    # Print the model architecture
    print(model)
