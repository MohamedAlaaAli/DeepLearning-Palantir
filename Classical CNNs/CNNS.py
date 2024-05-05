import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm



######################################################################################################################################################################################################################################################################################################### LeNet paper

class LeNet(nn.Module):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
    
    def forward(self, input):
        output = self.relu(self.conv1(input))
        output = self.pool(output)
        output = self.relu(self.conv2(output))
        output = self.pool(output)
        output = self.relu(self.conv3(output))
        output = output.reshape(output.shape[0], -1)
        output = self.relu(self.linear1(output))
        output = self.linear2(output)
        return output

    def test_lenet():
        input = torch.randn(64, 1, 32, 32)
        model = LeNet()
        return model(input)

######################################################################################################################################################################################################################################################################################################### AlexNet paper
class VGG16(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG16,self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifer(x)
        return x
    

#model = VGG16(in_channels=3, num_classes=1000)
#x = torch.randn(3, 3, 224, 224)
#print(model(x).shape)

################################################################################################################################################################################################################################################################# Inception paper

class Google_net(nn.Module):
    def __init__(self,in_channels=3, n_classes=1000, aux_classifiers=True):
        super(Google_net, self).__init__()
        self.in_channels = in_channels
        self.aux_classifiers = aux_classifiers
        self.conv1 = conv_block(in_channels=self.in_channels,
                               out_channels=64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = conv_block(in_channels=64,
                               out_channels=192,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        #self, in_channels, out_1x1, red_3x3, out_3x3,
        #         red_5x5, out_5x5, out_1x1pool   from the original paper
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, n_classes)

        if aux_classifiers:
            self.aux1 = Auxil_classifier(512, n_classes)
            self.aux2 = Auxil_classifier(528, n_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x= self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        # Auxiliary Softmax classifier 1
        if self.aux_classifiers and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_classifiers and self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = self.dropout(x)
        x = self.linear(x.reshape(x.shape[0], -1))

        if self.aux_classifiers and self.training:
            return aux1, aux2, x
        else:
            return x


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3,
                 red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3),stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )
    
    def forward(self, x): #N x C x H x W
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class Auxil_classifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Auxil_classifier, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = conv_block(in_channels=self.in_channels, out_channels=128, kernel_size=(1, 1))
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


#x = torch.rand(3, 3, 224, 224)
#model = Google_net(in_channels=3, n_classes=1000, )
#print(model(x)[2].shape)

######################################################################################################################################################################################################################################################################################################### ResNet paper
 
