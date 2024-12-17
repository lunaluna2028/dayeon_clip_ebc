import torch
from blocks import split_into_blocks
from graph import create_graph
def train_one_epoch(model, optimizer, train_loader, device, backbone):
    model.train()
    total_loss = 0

    for image, label in train_loader:
        # Tensor → NumPy 변환
        image = image.squeeze(0).permute(1, 2, 0).numpy()  # (B, C, H, W) → (H, W, C)
        label = label.numpy()

        # 블록 분할
        blocks, block_labels = split_into_blocks(image, label, block_size=64)

        # NumPy → Tensor 변환
        blocks_tensor = torch.stack([
            torch.tensor(block.transpose(2, 0, 1), dtype=torch.float32) for block in blocks
        ]).to(device)

        # Backbone 모델로 특징 추출
        block_features = backbone(blocks_tensor)

        # 그래프 생성
        graph = create_graph(block_features, block_labels, block_size=64, image_shape=image.shape)

        # 학습
        optimizer.zero_grad()
        output = model(graph)
        loss = F.mse_loss(output, graph.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, test_loader, device, backbone):
    model.eval()
    total_mae = 0
    total_mse = 0

    with torch.no_grad():
        for image, label in test_loader:
            # Tensor → NumPy 변환
            image = image.squeeze(0).permute(1, 2, 0).numpy()  # (B, C, H, W) → (H, W, C)
            label = label.numpy()  # Tensor → NumPy

            # 블록 분할 및 라벨 생성
            blocks, block_labels = split_into_blocks(image, label, block_size=64)

            # NumPy → Tensor 변환
            blocks_tensor = torch.stack([
                torch.tensor(block.transpose(2, 0, 1), dtype=torch.float32) for block in blocks
            ]).to(device)

            # Backbone 모델로 특징 추출
            block_features = backbone(blocks_tensor)

            # 그래프 생성
            graph = create_graph(block_features, block_labels, block_size=64, image_shape=image.shape)

            # 모델 예측
            output = model(graph)
            predicted_total = output.sum().item()
            actual_total = graph.y.sum().item()

            # MAE, MSE 계산
            total_mae += abs(predicted_total - actual_total)
            total_mse += (predicted_total - actual_total) ** 2

    mae = total_mae / len(test_loader)
    mse = total_mse / len(test_loader)
    return mae, mse
