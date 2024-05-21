# Authors: Melisa Mete (150200316),
#          Öykü Eren (150200326),
#          Bora Boyacıoğlu (150200310)

"""
Custom class for FaceDataset.
"""

# Import necessary libraries.
import numpy as np

import torch
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, images_path, landmarks_path, labels_path, transform=None):
        self.images = np.load(images_path).astype(np.float32)
        self.landmarks = np.load(landmarks_path).astype(np.float32)
        self.labels = np.load(labels_path)
        self.transform = transform
    
    def size(self):
        return self.images.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        landmarks = self.landmarks[idx]
        label = self.labels[idx]
        image = image.transpose(2, 0, 1)
        # Transform the image to match the expected input shape [C, H, W]
        if self.transform:
            image = self.transform(image)

        # Convert landmarks to a flat tensor
        landmarks = torch.tensor(landmarks, dtype=torch.float32).view(-1)
        label = torch.tensor(label, dtype=torch.long)
        return image, landmarks, label