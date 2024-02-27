import json
import os

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
import os


class PRMIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, min_zoom=0.75, grayscale_mask=True):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = v2.Compose([
            v2.RandomResizedCrop(160, scale=(min_zoom, 1.0),
                                 ratio=(1, 1), antialias=None),
            v2.RandomHorizontalFlip(p=0.5)
        ])
        self.grayscale_mask = grayscale_mask
        self.img_filenames = self.get_all_filenames()

    def get_all_filenames(self) -> list:
        all_files = []
        for root, _, files in os.walk(self.img_dir):
            filenames = [os.path.join(root, file) for file in files]
            all_files.extend(filenames)

        return all_files

    def __len__(self) -> int:
        return len(self.img_filenames)

    def get_image(self, filename) -> np.ndarray:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_mask(self, filename) -> np.ndarray:
        filename = os.path.join(os.path.dirname(
            filename), "GT_" + os.path.basename(filename))
        filename = filename.replace(
            self.img_dir, self.mask_dir).replace('.jpg', '.png')

        mask = cv2.imread(filename)
        return mask

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        image = self.get_image(self.img_filenames[index])
        mask = self.get_mask(self.img_filenames[index])

        image = F.to_image(image)
        mask = F.to_image(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = F.resize(image, 160, antialias=None)
            mask = F.resize(mask, 160, antialias=None)

        if self.grayscale_mask:
            mask = F.to_grayscale(mask)

        image = F.to_dtype(image, torch.float32, scale=True)
        mask = F.to_dtype(mask, torch.float32, scale=True)

        return {'image': image, 'mask': mask}
