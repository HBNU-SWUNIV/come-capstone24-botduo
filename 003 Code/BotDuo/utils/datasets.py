import os
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class StegDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        image = Image.open(image_path).convert('RGB')
        
        if image.mode == 'L':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image=np.array(image, dtype=np.float32))['image']
            
        return image, label

class BenchmarkDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        
        image = np.array(image, dtype=np.float32)

        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label