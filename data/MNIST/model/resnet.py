from torch import nn
from torch.nn import functional as F
from optim import OptimRegime

from utils.args import parse_args
args = parse_args()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # self.bn = nn.BatchNorm2d(channels, track_running_stats=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv1(y)
        return F.relu(x + y)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.rblock1 = ResidualBlock(channels=16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.rblock2 = ResidualBlock(channels=32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

        self.regime = [{'epoch': 0,
                'optimizer': 'SGD',
                'lr': args.lr_ResNet,
                'momentum': 0.5}]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = OptimRegime(self.parameters(), self.regime)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
