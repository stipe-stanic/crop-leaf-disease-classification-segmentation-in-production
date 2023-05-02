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

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(p = 0.5)

        self.fc3 = nn.Linear(512, 4)

    def forward(self, x):
        # Konvolucijski slojevi
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Linearni slojevi
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    # Kod Cifara model vraća otprilike 80% točnosti u 30 epoha. 87,75% nakon 200 epoha. Rekord je inače preko 99%. ;)
    # Namjerno je ostao praktički identičan model kao i kod MNIST-a da vidite koliko jako ovisi dataset na treniranje.
    # Promjenjen je samo ulaz i broj parametara u prvi linearni sloj.
