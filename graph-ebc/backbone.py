import torch.nn as nn
from torchvision.models import resnet18

class BackboneModel(nn.Module):
    def __init__(self):
        super(BackboneModel, self).__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 마지막 두 레이어 제거
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 512  # ResNet18의 출력 채널

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (B, D)
