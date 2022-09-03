from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch import flatten


class Net(Module):
    def __init__(self, numChannels, classes):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.bn1 = BatchNorm2d(32)
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = Dropout(0.25)

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.bn2 = BatchNorm2d(64)
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = Dropout(0.25)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.relu3 = ReLU()
        self.bn3 = BatchNorm2d(128)
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = Dropout(0.25)

        self.fc1 = Linear(in_features=6144, out_features=500)
        self.relu4 = ReLU()
        self.bn4 = BatchNorm1d(500)
        self.dropout4 = Dropout(0.5)

        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output
