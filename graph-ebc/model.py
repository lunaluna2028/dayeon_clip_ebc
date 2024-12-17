import torch.nn as nn
from torch_geometric.nn import GCNConv

class GraphEBCModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEBCModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = nn.functional.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze()
