import os
import cv2
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

class UCFCrowdDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".mat")])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 로드
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 라벨 로드
        label = loadmat(self.label_paths[idx])["annPoints"]  # (N, 2)
        
        # NumPy 배열 반환
        return image, label