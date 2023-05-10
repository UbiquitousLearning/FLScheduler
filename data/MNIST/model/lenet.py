import torch.nn as nn
import torch.nn.functional as F
from optim import OptimRegime
import torch

from utils.args import parse_args
args = parse_args()

class lenet(nn.Module):
    '''
    An example model for mnist from pytorch tutorial
    float_model
    '''
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.regime = [{'epoch': 0,
                        'optimizer': 'SGD',
                        'lr': args.lr_LeNet,
                        'momentum': 0.5}]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = OptimRegime(self.parameters(), self.regime)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
