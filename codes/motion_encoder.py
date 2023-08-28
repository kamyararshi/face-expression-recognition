import torch
import torch.nn as nn
import torchvision.models as models

### The ResNet is trained from scratch for the facial expresion prediction
class ResNet18(nn.Module):
    def __init__(self, in_channels, pretrained=False):
        super(ResNet18, self).__init__()
        # If we want pretrained weights or not
        if pretrained:
            self.model = models.resnet18(weights=pretrained)
        else:
            self.model = models.resnet18(weights=False)
        # in_channels    
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Set last fully connected layer to output 7 numbers;
        # 3 for rotation, 3 for translation and 1 for scale (also needed to build translation matrix afterwards)
        self.model.fc = nn.Linear(512, 7)

    def forward(self, x):
        return self.model(x)
