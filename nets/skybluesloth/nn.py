import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv = nn.Conv2d(1, conv_out_channels, conv_kernel_size, conv_stride)
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        self.head = nn.Linear(lin_in_features, lin_out_features)
        self.emb = nn.Linear(1, embedding_length)
        self.final = nn.Linear(head_in_features, 9)

    def forward(self, grid, heading):
        grid = F.leaky_relu(self.conv(grid))
        grid = self.pool(grid)
        grid = grid.view(lin_in_features, -1).transpose(0,1)
        grid = F.leaky_relu(self.head(grid))
        embedding = F.leaky_relu(self.emb(heading))
        conc = torch.cat((grid, embedding), 1)
        return self.final(conc)
