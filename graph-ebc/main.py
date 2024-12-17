import torch
from torch.utils.data import DataLoader
from backbone import BackboneModel
from model import GraphEBCModel
from train import train_one_epoch, evaluate
from dataset import UCFCrowdDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = BackboneModel().to(device)
model = GraphEBCModel(in_channels=backbone.out_channels, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = UCFCrowdDataset("/root/crowd_counting/data/UCF-QNRF/Train", "/root/crowd_counting/data/UCF-QNRF/Train")
test_dataset = UCFCrowdDataset("/root/crowd_counting/data/UCF-QNRF/Test", "/root/crowd_counting/data/UCF-QNRF/Test")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for epoch in range(1, 101):
    print(f"Epoch {epoch}")
    train_loss = train_one_epoch(model, optimizer, train_loader, device, backbone)
    print(f"Train Loss: {train_loss:.4f}")
    if epoch >= 50:
        mae, mse = evaluate(model, test_loader, device, backbone)
        print(f"Evaluation - MAE: {mae:.4f}, MSE: {mse:.4f}")
