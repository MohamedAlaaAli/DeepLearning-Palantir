import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchsummary import summary 

#------------------------------------------------------------------------------------- Utility Classes -----------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, f_1x1, f_3x3_adaptor, f_3x3, f_5x5_adaptor, f_5x5, f_mp_adaptor):
        """
        f_1x1: number of filters for 1x1 conv.
        f_3x3_adaptor: number of 1x1 filters before 3x3 filters.
        f_3x3: number of filters for 3x3 conv.
        f_5x5_adaptor: number of 1x1 filters before 5x5 filters.
        f_5x5: number of filters for 5x5 conv
        f_mp_adaptor: number of 1x1 filters after max pooling.
        
        """
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_1x1, kernel_size=1,stride=1, padding=0)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_adaptor, kernel_size=1,stride=1, padding=0), 
            ConvBlock(in_channels, f_3x3, kernel_size=3,stride=1, padding=1) 
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, f_5x5_adaptor, kernel_size=1, stride = 1, padding=0),
            ConvBlock(in_channels, f_5x5, kernel_size=5,stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride = 1, padding=1, ceil_mode=True), # preserve the spatial dimension for concatenation.
            ConvBlock(in_channels, f_mp_adaptor, kernel_size = 1, stride = True, padding = 0)
        )

        def forward(self, x):
            branch1 = self.branch1(x)
            branch2 =  self.branch2(x)
            branch3 =  self.branch3(x)
            branch4 =  self.branch4(x)

            return torch.cat([branch1, branch2, branch3, branch4], 1) # concatenate in the channel dim
    

class AuxClassifier():
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((4,4)) # (4,4) regardless the input spatial dimensions 
        self.conv = nn.Conv2d(in_channels, 128, kernel_size= 1, stride= 1, padding=0)
        self.act1 = nn.ReLU()
        nn.fc1 = nn.Linear(2048, 1024)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def farward(self,x):
        x = self.pool(x)

        x = self.conv(x)
        x = self.act1(x)

        x = nn.fc1(x)
        x = self.act2(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x 

#------------------------------------------------------------------------------------- Utility Classes -----------------------------------------------------------------

class GoogleNet(nn.Module):

    def __init__(self, num_classes = 10):
        super(GoogleNet, self).__init__()

        self.conv1 = ConvBlock(in_channels = 3, out_channels = 64, kernel_size = 7, stride=2, padding=3)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2, padding=0, ceil_mode=True)

        # layer of depth 2 (depth here refers the max number of convolutional layers in-between input and output)
        # meaning that the (adaptor + conv) layer collectively represent in the architecture table. 
        self.conv2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride = 2, padding= 0, ceil_mode=True)

        # layer of depth 2  
        self.inception3A = InceptionModule(in_channels = 192,
                                            f_1x1 = 64,
                                            f_3x3_adaptor = 96,
                                            f_3x3 = 128,
                                            f_5x5_adaptor = 16,
                                            f_5x5 = 32,
                                            f_mp_adaptor = 32)
        
        self.inception3B = InceptionModule(in_channels = 256,
                                            f_1x1 = 128,
                                            f_3x3_adaptor = 128,
                                            f_3x3 = 192,
                                            f_5x5_adaptor = 32,
                                            f_5x5 = 96,
                                            f_mp_adaptor = 64)
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride = 2, padding=0, ceil_mode=True)

        self.inception4A = InceptionModule(in_channels = 480,
                                            f_1x1 = 192,
                                            f_3x3_adaptor = 96,
                                            f_3x3 = 208,
                                            f_5x5_adaptor = 16,
                                            f_5x5 = 48,
                                            f_mp_adaptor = 64)
        
        self.inception4B = InceptionModule(in_channels = 512,
                                            f_1x1 = 160,
                                            f_3x3_adaptor = 112,
                                            f_3x3 = 224,
                                            f_5x5_adaptor = 24,
                                            f_5x5 = 64,
                                            f_mp_adaptor = 64)
        
        self.inception4C = InceptionModule(in_channels = 512,
                                            f_1x1 = 128,
                                            f_3x3_adaptor = 128,
                                            f_3x3 = 256,
                                            f_5x5_adaptor = 24,
                                            f_5x5 = 64,
                                            f_mp_adaptor = 64)
                
        self.inception4D = InceptionModule(in_channels = 512,
                                            f_1x1 = 112,
                                            f_3x3_adaptor = 144,
                                            f_3x3 = 288,
                                            f_5x5_adaptor = 32,
                                            f_5x5 = 64,
                                            f_mp_adaptor = 64)
        
        self.inception4E = InceptionModule(in_channels = 528,
                                            f_1x1 = 256,
                                            f_3x3_adaptor = 160,
                                            f_3x3 = 320,
                                            f_5x5_adaptor = 32,
                                            f_5x5 = 128,
                                            f_mp_adaptor = 128)
        
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride = 2, padding=0,ceil_mode=True)
        
        self.inception5A = InceptionModule(in_channels = 832,
                                            f_1x1 = 256,
                                            f_3x3_adaptor = 160,
                                            f_3x3 = 320,
                                            f_5x5_adaptor = 32,
                                            f_5x5 = 128,
                                            f_mp_adaptor = 128)
        
        self.inception5B =  InceptionModule(in_channels = 832,
                                            f_1x1 = 384,
                                            f_3x3_adaptor = 192,
                                            f_3x3 = 384,
                                            f_5x5_adaptor = 48,
                                            f_5x5 = 128,
                                            f_mp_adaptor = 128)
        
        self.pool5 = nn.AdaptiveAvgPool1d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)

        # auxiliary classifiers used for training 

        self.Aux1 = AuxClassifier(in_channels=512, num_classes = num_classes) # num_channels is the output of the 3rd Inception module (4A) 
        self.Aux2 = AuxClassifier(in_channels=528, num_classes = num_classes) # num_channels is the output of the 6th Inception module (4D)

        def forward(self, x):
            # the stem 
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.pool2(x)

            # consecutive Inception blocks (3's family) 
            x = self.inception3A(x)
            x = self.inception3B(x)


            # consecutive Inception blocks (4's family)
            x = self.inception4A(x)
# branching
            # auxiliary classifier branch
            aux1_output = self.Aux1(x)

            x = self.inception4B(x)
            x = self.inception4C(x)
            x = self.inception4D(x)
# branching
            # auxiliary classifier branch
            aux2_output = self.Aux2(x)

            x = self.inception4E(x)


            # consecutive Inception blocks (5's famity)
            x = self.inception5A(x)
            x = self.inception5B(x)


            x = self.pool5(x)
            x = torch.flatten(x,1) 
            x = self.dropout(x)
            x = self.fc(x)

            return x, aux1_output, aux2_output











        

