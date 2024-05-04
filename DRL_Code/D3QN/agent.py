import torch.nn as nn
import torch

class Block(nn.Module):
    """
    A residual block used in the creation of a CNN.

    Attributes:
        res (bool): If True, includes a residual connection.
        left (nn.Sequential): Sequential model for the main pathway.
        shortcut (nn.Sequential): Sequential model for the shortcut pathway.
        relu (nn.Sequential): ReLU activation applied after adding the shortcut.
    """
    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
        )
        # Shortcut to match dimensions if necessary
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out

class MyDQN(nn.Module):
    """
    The neural network architecture for the Dueling Double DQN (D3QN).

    Attributes:
        cfg (list): Configuration for channel sizes in the CNN layers.
        res (bool): If True, enables residual connections in blocks.
        features1 (nn.Sequential): Sequential model of convolutional blocks.
        A_1, A_2, V_1, V_2 (nn.Linear): Layers for advantage and value streams.
        flatten (nn.Flatten): Flattens the output of the conv layers to feed into linear layers.
        relu (nn.ReLU): ReLU activation function.
    """
    def __init__(self, cfg=[8, 8, 16], res=True):
        super(MyDQN, self).__init__()
        self.res = res
        self.cfg = cfg
        self.inchannel1 = 8
        self.features1 = self.make_layer()

        # Advantage stream
        self.A_1 = nn.Linear(160, 32)
        self.A_2 = nn.Linear(32, 2)

        # Value stream
        self.V_1 = nn.Linear(160, 32)
        self.V_2 = nn.Linear(32, 1)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self):
        """
        Creates the layers of the network based on configuration.
        """
        layers = []
        layers.append(Block(self.inchannel1, self.cfg[0], self.res))
        layers.append(nn.Conv2d(self.cfg[0], self.cfg[0], kernel_size=2, padding=0, stride=2, bias=False))
        layers.append(Block(self.cfg[0], self.cfg[1], self.res))
        layers.append(nn.Conv2d(self.cfg[1], self.cfg[1], kernel_size=2, padding=0, stride=2, bias=False))
        layers.append(Block(self.cfg[1], self.cfg[2], self.res))
        layers.append(nn.Conv2d(self.cfg[2], self.cfg[2], kernel_size=2, padding=0, stride=2, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Creates the layers of the network based on configuration.
        """
        x = self.features1(x)
        x = self.flatten(x)

        # Compute value stream
        V = self.V_1(x)
        V = self.relu(V)
        V = self.V_2(V)

        # Compute advantage stream
        A = self.A_1(x)
        A = self.relu(A)
        A = self.A_2(A)

        # Combine streams into Q-values
        x_out = V.expand_as(A) + (A - A.mean().expand_as(A))
        return x_out


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    seed_value = 0
    torch.manual_seed(seed_value)
    model = MyDQN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"The number of parameters of agent is : {total_params}")