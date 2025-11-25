import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x 
class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name. 
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model does not achieve the required parameter count (101700).
    """
    def __init__(self, depth=20, dim=32, input_size=28*28, kernel_size = 3, patch_size = 9, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.bn1 = nn.BatchNorm2d(dim)
        
        self.convmixlayers = nn.Sequential(*[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],)
        
        #self.convmixlayers = nn.Sequential[*layers]
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(dim, num_classes)

    def forward(self, x):
        # The forward pass defines the connectivity of the layers defined in __init__.
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        x = self.convmixlayers(x)
        x = self.avgpool1(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        return x
