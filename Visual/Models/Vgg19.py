import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):
    def __init__(self, num_convolutions, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Convolution, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convolutions = nn.ModuleList(
            [nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=kernel_size, stride=stride, padding=padding ) for _ in range(num_convolutions)]
        )
    
    def forward(self, x):
        x = self.layer1(x)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        return x

class VGG19(nn.Module):
    def __init__(self, flatten_output, num_classes = 1000):
        super(VGG19, self).__init__()
        self.convolution_layers = nn.Sequential(
            Convolution(2, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Convolution(2, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Convolution(2, in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # No. of convolutions Changed for Tiny-imagenet
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution(2, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # No. of convolutions Changed for Tiny-imagenet
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution(2, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # No. of convolutions Changed for Tiny-imagenet
            # nn.MaxPool2d(kernel_size=2, stride=2) # Changed for Tiny-imagenet
        )
    
        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_output, 4096),    #flatten_output: 7*7*512
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.convolution_layers(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x