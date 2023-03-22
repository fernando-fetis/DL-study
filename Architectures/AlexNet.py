import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1000):
        super().__init__()

        # Convolutional layers:
        self.conv1 = nn.Conv2d(in_channels, 96, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)

        # Fully connected layers:
        self.dense1 = nn.Linear(6*6*256, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, n_classes)

    def forward(self, x):  # net input: Cx227x227 image.
        x = nn.ReLU()(self.conv1(x))
        x = nn.LocalResponseNorm(5)(x)
        x = nn.MaxPool2d(3, stride=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.LocalResponseNorm(5)(x)
        x = nn.MaxPool2d(3, stride=2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = nn.MaxPool2d(3, stride=2)(x)
        x = nn.Flatten()(x)
        x = nn.ReLU()(self.dense1(x))
        x = nn.Dropout(0.5)(x)
        x = nn.ReLU()(self.dense2(x))
        x = nn.Dropout(0.5)(x)
        x = self.dense3(x)
        return x