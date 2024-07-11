import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(256 * 6* 6, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.shape[0], -1) #flatten the output
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out