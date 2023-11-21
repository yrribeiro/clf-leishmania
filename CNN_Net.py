import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    Convolutional Neural Net structure designed for extracting leishmania features in images
    '''
    def __init__(self, embedding_size, IMG_SIZE):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc_input_size = self.calculate_fc_input_size(IMG_SIZE)
        self.fc1 = nn.Linear(self.fc_input_size, embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def calculate_fc_input_size(self, input_size):
        x = torch.randn(1, *input_size)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x.size(1)