import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    Convolutional Neural Net designed for extracting leishmania features in images.
    '''
    def __init__(self, embedding_size, IMG_SIZE):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.25)

        fc_input_size = self.calculate_conv_output_size(IMG_SIZE)

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def calculate_conv_output_size(self, input_size):
        x = torch.randn(1, *input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        return x.size(1)