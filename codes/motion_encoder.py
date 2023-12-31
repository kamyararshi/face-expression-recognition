import torch
import torch.nn as nn
import torchvision.models as models

### The ResNet is trained from scratch for the facial expresion prediction
class ResNet18(nn.Module):
    def __init__(self, in_channels, out_classes, pretrained=False):
        super(ResNet18, self).__init__()
        # If we want pretrained weights or not
        if pretrained:
            self.model = models.resnet18(weights=pretrained)
        else:
            self.model = models.resnet18(weights=False)
        # in_channels    
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Set last fully connected layer to output 8 numbers;
        # due to 8 emotions as labels for ck+
        # TODO: find the best architecture for emotic
        self.model.fc = nn.Linear(512, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.AdaptiveAvgPool1d(64),
            nn.ReLU(),
            nn.Linear(64, out_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

