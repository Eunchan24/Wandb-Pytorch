from distutils.command import config
import torch.nn as nn
import torch.nn.functional as F
class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.config = config 
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.config.conv1_channels, 5), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.config.conv1_channels, self.config.conv2_channels,  5), nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Linear(self.config.conv2_channels * 5 * 5, self.config.hidden_nodes), nn.ReLU(),
            nn.Dropout2d(p=self.config.dropout))
        self.layer4 = nn.Sequential(
            nn.Linear(self.config.hidden_nodes, 84), nn.ReLU(),
            nn.Dropout2d(p=self.config.dropout))
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.config.conv2_channels * 5 * 5)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc3(x)
        return x