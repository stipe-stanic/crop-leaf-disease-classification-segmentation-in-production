import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_01(nn.Module):

    def __init__(self):
        super(Model_01, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res1 = nn.Sequential(nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        )

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res2 = nn.Sequential(nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        )

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res3 = nn.Sequential(nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        )

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res4 = nn.Sequential(nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ), nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        )

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.bn10 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn11 = nn.BatchNorm1d(512)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.maxpool1(x)

        x = self.res1(x) + x

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.maxpool2(x)

        x = self.res2(x) + x

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.maxpool3(x)

        x = self.res3(x) + x

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.maxpool4(x)

        x = self.res4(x) + x

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.maxpool5(x)

        # Linear layers
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn10(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
