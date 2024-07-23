"""
Using ResNet as the backbone network to create the in-network feature pyramid.

"""

import torch 
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4 # global for the bottleneck blocks
    def __init__(self, in_channels, in_between_channels, stride = 1 ):
        super(Bottleneck, self).__init__()

        # bottleneck block
        self.conv1 = nn.Conv2d(in_channels, in_between_channels, kernel_size=1, bias=False) # bias is redundant due to subsequent BN
        self.bn1 = nn.BatchNorm2d(in_between_channels)
        self.conv2 = nn.Conv2d(in_between_channels,in_between_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_between_channels)
        self.conv3 = nn.Conv2d(in_between_channels, in_between_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_between_channels * self.expansion) 

        self.shortcut = nn.Sequential() # by default no dimensionality matching required to conduct the shortcut due to the lack of (spatial dim variation, channels mismatch)
        # dimensionality matching
        if stride != 1 or in_channels != self.expansion * in_between_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_between_channels * self.expansion, kernel_size= 1, stride=stride, bias=False),
                nn.BatchNorm2d(in_between_channels * self.expansion)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)

        out = nn.functional.relu(out)

        return out 
    


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3,64, kernel_size=7, stride = 2, padding = 3, bias = False)
        self.bn1  = nn.BatchNorm2d(64)

        # bottom-up pathway
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 

        # smoothing filter for reducing the effect of aliasing of upsampling in the top-down pathway
        self.smooth = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1) # one for all levels of feature maps

        self.top_layer = nn.Conv2d(2048, 256, kernel_size=1, stride =1, padding=0) # reduce no. channels

        # used to unify the channels of each feature map level
        self.lateral_layer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  
        self.lateral_layer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = list()
        for st in strides:
            layers.append(block(self.in_channels, channels, st))
            self.in_channels = block.expansion * channels   # update the in_channels for subsequent blocks
        return nn.Sequential(*layers)  

    def _upsample_add(self, x, y):
        """
        x: top feature map to be upsampled.
        y: lateral feature map to be added.
    
        """
        _,_,h,w = y.size()
        return nn.functional.upsample(x, size=(h,w), mode = "bilinear") + y # using bilinear to specify the exact size 
    
    def forward(self, x):
        # bottom-up pathway 
        c1 = nn.functional.relu(self.bn1(self.conv1(x)))
        c1 = nn.functional.max_pool2d(c1, kernel_size=3, stride=2, padding=1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # top-down pathway 
        p5 = self.top_layer(c5) # top-most feature map 
        p4 = self._upsample_add(p5, self.lateral_layer1(c4))
        p4 = self.smooth(p4) # this in-between smoothing prevents propagating the aliasing resulting from upsampling 
        p3 = self._upsample_add(p4, self.lateral_layer2(c3))
        p3 = self.smooth(p3)
        p2 = self._upsample_add(p3, self.lateral_layer3(c2))
        p2 = self.smooth(p2)
        return p2, p3, p4, p5
    


def FPN_ResNet_101():
    return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN_ResNet_101()
    fms = net(torch.randn(1,3,600,900))
    for fm in fms:
        print(fm.size())

test()



















