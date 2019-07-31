import torch
import torch.nn as nn
from torchvision.models import resnet101


class ResNetBackbone(nn.Module):
    def __init__(self, no_outputs):
        super(ResNetBackbone, self).__init__()
        self.pretrained = resnet101(pretrained=True)

        self.last_convs = nn.Sequential(
            nn.Conv2d(512, 512, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 1024, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, (3, 3)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 1 * 13, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, no_outputs)
        )

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        #c3 = self.pretrained.layer3(c2)
        x = self.last_convs(c2)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x