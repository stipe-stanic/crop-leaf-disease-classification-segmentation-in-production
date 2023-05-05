import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with two 3x3 convolutional layers followed by batch normalization layers and max pooling."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

    def forward(self, x):
        """Applies the convolutional block to the input tensor x."""

        return self.conv_block(x)


class ResidualBlock(nn.Module):
   """Residual block with two convolutions followed by batch normalization layers"""

   def __init__(self, in_channels: int, hidden_channels: int,  out_channels: int):
       super().__init__()
       self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
       self.bn1 = nn.BatchNorm2d(hidden_channels)

       self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
       self.bn2 = nn.BatchNorm2d(out_channels)

   def residual_block(self, x):
       x1 = F.relu(self.bn1(self.conv1(x)))
       x2 = F.relu(self.bn2(self.conv2(x1)))
       return x2 + x

   def forward(self, x): return self.residual_block(x)


class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, after_conv: bool = False):
        super().__init__()

        if after_conv:
            self.linear = nn.Linear(in_channels * 14 * 14, out_channels)
        else:
            self.linear = nn.Linear(in_channels, out_channels)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = self.linear
        self.bn1 = nn.BatchNorm1d(out_channels)

    def linear_block(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return x

    def forward(self, x): return self.linear_block(x)


class ResModel(nn.Module):

    def __init__(self):
        super(ResModel, self).__init__()

        self.conv1 = ConvBlock(3, 32, 32)
        self.res1 = ResidualBlock(32, 32, 32)

        self.conv2 = ConvBlock(32, 64, 64)
        self.res2 = ResidualBlock(64, 64, 64)

        self.conv3 = ConvBlock(64, 128, 128)
        self.res3 = ResidualBlock(128, 128, 128)

        self.conv4 = ConvBlock(128, 256, 256)
        self.res4 = ResidualBlock(256, 256, 256)

        self.fc1 = LinearBlock(256, 512, after_conv=True)
        self.fc2 = LinearBlock(512, 512)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # Conv blocks
        x = self.conv1(x)
        x = self.res1(x)

        x = self.conv2(x)
        x = self.res2(x)

        x = self.conv3(x)
        x = self.res3(x)

        x = self.conv4(x)
        x = self.res4(x)

        # Linear blocks
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.fc2(x)

        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
