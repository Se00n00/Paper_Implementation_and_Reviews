import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, config.conv1["out_channels"], kernel_size=11, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(config.conv1["out_channels"], alpha=0.0001, beta=0.75, k=2),
            
            nn.Conv2d(config.conv1["out_channels"], config.conv2["out_channels"], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(config.conv1["out_channels"], alpha=0.0001, beta=0.75, k=2),
            
            nn.Conv2d(config.conv2["out_channels"], config.conv3["out_channels"], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(config.conv3["out_channels"], config.conv4["out_channels"], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(config.conv4["out_channels"], config.conv5["out_channels"], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(config.classifer_in, config.classifer_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.classifer_hidden),
            nn.Dropout(p=0.5),
            nn.Linear(config.classifer_hidden, config.classifer_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.classifer_hidden, config.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x