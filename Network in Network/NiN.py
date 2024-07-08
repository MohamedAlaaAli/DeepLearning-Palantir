import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkInNetwork(nn.Module):

    def __init__(self, num_classes, input_dim):
        super(NetworkInNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=96, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1)  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=1)   
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=1)   
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1)   
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=1)   
        self.conv9 = nn.Conv2d(in_channels=96, out_channels=num_classes, kernel_size=1)   

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.conv9(x)

        
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
        return F.log_softmax(x.squeeze(-1).squeeze(-1), dim=1)

num_classes = 10  
input_dim = 3     
batch_size = 1
height, width = 28, 28  
x = torch.randn(batch_size, input_dim, height, width)
model = NetworkInNetwork(num_classes, input_dim)
print("Input shape:", x.shape)
output = model(x)
print("Output shape:", output.shape)