import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Residual(nn.Module):
    """
    Residual connection, sums together transformation and input.
    """
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
    #Model default parameters set to hyperparameter optimal results
    def __init__(self, depth=2, dim=23, coarse_depth = 2, fc_size = 32, kernel_size = 3, num_classes=10):
        super(Net, self).__init__()
        
        #Initialise depth and coarse_depth attributes to check to initialise and 
        #do forward pass for deep convolutional layer and coarsening layer
        self.depth = depth
        self.coarse_depth = coarse_depth

        #Input layer
        self.conv1 = Residual(nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = dim, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(dim),
            nn.ReLU()
            ))
        
        #Deep convolutional layer
        if depth:
            self.deepconv = nn.Sequential(*[Residual(nn.Sequential(
                nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size=kernel_size, padding="same"),
                nn.BatchNorm2d(dim),
                nn.ReLU()
                ))
                for _ in range(depth)])
        
        #Coarsening layer
        if coarse_depth:
            self.coarse = nn.Sequential(*[nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size=kernel_size, padding="same"),
                nn.BatchNorm2d(dim),
                nn.ReLU())
                for _ in range(coarse_depth)])
        
        #Average pool operator to feed to FC layer
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))

        #Fully connected layer
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=dim, out_features=fc_size),
            nn.ReLU()
        )

        self.output = nn.Linear(in_features=fc_size, out_features=num_classes)

    def forward(self, x):
        # The forward pass defines the connectivity of the layers defined in __init__.
        x = self.conv1(x) #Input layer
    
        #Deep Convolutional layer
        if self.depth:
            x = self.deepconv(x)

        #Coarsening layer
        if self.coarse_depth:
            x = self.coarse(x)

        
        x = self.avgpool1(x) #Average pool operator to feed to FC layer
        x = self.fc1(x) #Fully connected layer
        x = self.output(x) #Output layer

        return x
