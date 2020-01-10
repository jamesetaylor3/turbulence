import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.head = nn.Linear(128, 24)
        self.emb = nn.Linear(1, embedding_length)
        self.final = nn.Linear(30, 9)

    def forward(self, grid, heading):
        grid = F.leaky_relu(self.conv1(grid))
        grid = self.pool(grid)
        grid = F.leaky_relu(self.conv2(grid))
        grid = self.pool(grid)
        grid = grid.view(128, -1).transpose(0,1)
        grid = F.leaky_relu(self.head(grid))
        embedding = F.leaky_relu(self.emb(heading))
        conc = torch.cat((grid, embedding),1)
        return self.final(conc)
