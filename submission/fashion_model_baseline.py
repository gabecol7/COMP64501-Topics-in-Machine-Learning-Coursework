import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name. 
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model does not achieve the required parameter count (101700).
    """
    def __init__(self, input_size=28*28, hidden_size=16, num_classes=10):
        super(Net, self).__init__()

        # We define the layers of our model here by instantiating layer objects.
        # Here we define two fully connected (linear) layers.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.out = nn.Linear(in_features=64*2*2, out_features=10)

        # nn.init module contains initialization methods:
        # https://docs.pytorch.org/docs/main/nn.init.html
        # This particular one is called Kaiming initialization (also known as
        # He initialization) described in He, K. et al. (2015)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self, x):
        # The forward pass defines the connectivity of the layers defined in __init__.
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = x.reshape(-1, 64*2*2)
        x = self.out(x)
        return x
