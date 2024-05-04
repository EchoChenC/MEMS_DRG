import torch
from torch import nn

class Block(nn.Module):
    """ A basic building block for a convolutional neural network with optional residual connections. """
    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass of the block."""
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out

class Net(nn.Module):
    """ A network composed of several 'Block' layers optionally using residual connections. """
    def __init__(self, cfg=[8, 8, 16], res=True):
        super(Net, self).__init__()
        self.res = res
        self.cfg = cfg
        self.inchannel1 = 3
        self.features1 = self.make_layer1()

        self.f1 = nn.Linear(160, 32)
        self.f3 = nn.Linear(32, 1)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)

    def make_layer1(self):
        """Construct the sequence of blocks for the model."""
        layers = []
        layers.append(Block(self.inchannel1, self.cfg[0], self.res))
        layers.append(nn.Conv2d(self.cfg[0], self.cfg[0], kernel_size=2, padding=0, stride=2, bias=False))
        layers.append(Block(self.cfg[0], self.cfg[1], self.res))
        layers.append(nn.Conv2d(self.cfg[1], self.cfg[1], kernel_size=2, padding=0, stride=2, bias=False))
        layers.append(Block(self.cfg[1], self.cfg[2], self.res))
        layers.append(nn.Conv2d(self.cfg[2], self.cfg[2], kernel_size=2, padding=0, stride=2, bias=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Defines the computation performed at every call."""
        x = self.features1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.relu(x)
        x = self.f3(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_value = 0
    torch.manual_seed(seed_value)

    model = Net().to(device).to(device)
    g1 = torch.randint(0, 2, (1, 3, 40, 20)).to(device).float() # Example input: (batch, channel, size1, size2)

    outputs = model(g1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"The number of parameters is: {total_params}")
