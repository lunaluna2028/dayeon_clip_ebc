import torch
from torch_geometric.data import Data

def create_graph(block_features, block_labels, block_size, image_shape):
    num_blocks = len(block_features)
    h, w = image_shape[:2]
    rows = h // block_size
    cols = w // block_size

    x = torch.tensor(block_features, dtype=torch.float)
    y = torch.tensor(block_labels, dtype=torch.float)

    edge_index = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i != j:
                xi, yi = divmod(i, cols)
                xj, yj = divmod(j, cols)
                if abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
                    edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)
