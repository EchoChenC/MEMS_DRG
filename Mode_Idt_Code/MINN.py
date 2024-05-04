import torch
from torch import nn

class MyNet(nn.Module):
    """ A custom neural network model for mode identification. """

    def __init__(self, output):
        """
        Initialize the neural network components.
        Args:
        n_classes (int): The number of output classes or labels.
        """
        super(MyNet, self).__init__()

        # Branch 1 layers:
        self.c1_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3_1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, padding=2)
        self.s4_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s6_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c7_1 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # Branch 2 layers:
        self.c1_2 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.s2_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3_2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, padding=2)
        self.s4_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s6_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c7_2 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # Activation and other functional layers:
        self.Sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        # Final dense layers:
        self.f7 = nn.Linear(240, 84)
        self.output = nn.Linear(84, output)

    def forward(self, x_1, x_2):
        """ Defines the forward pass of the model using two inputs. """
        # Processing the first input through branch 1:
        x_1 = self.Sigmoid(self.c1_1(x_1))
        x_1 = self.s2_1(x_1)
        x_1 = self.Sigmoid(self.c3_1(x_1))
        x_1 = self.s4_1(x_1)
        x_1 = self.c5_1(x_1)
        x_1 = self.s6_1(x_1)
        x_1 = self.c7_1(x_1)
        x_1 = self.flatten(x_1)

        # Processing the second input through branch 2:
        x_2 = self.Sigmoid(self.c1_2(x_2))
        x_2 = self.s2_2(x_2)
        x_2 = self.Sigmoid(self.c3_2(x_2))
        x_2 = self.s4_2(x_2)
        x_2 = self.c5_2(x_2)
        x_2 = self.s6_2(x_2)
        x_2 = self.c7_2(x_2)
        x_2 = self.flatten(x_2)

        # Combining the outputs from both branches:
        x = torch.cat((x_1, x_2), 1)
        x = self.f7(x)
        x = self.Sigmoid(self.output(x))
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model and prepare the synthetic inputs:
    model = MyNet(output=1).to(device)
    x1 = torch.rand([16, 1, 56, 56]).to(device)  # Example input 1: (batch, channe, size, size)
    x2 = torch.rand([16, 1, 56, 56]).to(device)  # Example input 2: (batch, channe, size, size)

    # Perform a forward pass and count model parameters:
    y = model(x1, x2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"The number of parameters is: {total_params}")