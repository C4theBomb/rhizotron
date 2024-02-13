import json
import os

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from processing.utils.file_management import get_image_filenames


class LabelmeDataset(Dataset):
    def __init__(self, dataset_dir, min_zoom=0.5, grayscale_mask=True):
        self.transform = v2.Compose([
            v2.RandomResizedCrop(400, scale=(min_zoom, 1.0), ratio=(1, 1), antialias=None),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])
        self.grayscale_mask = grayscale_mask
        self.img_filenames = get_image_filenames(dataset_dir, recursive=True)

    def __len__(self):
        return len(self.img_filenames)

    def get_image(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image[:image.shape[0] - 2, :image.shape[1]]

        return image

    def get_mask(self, filename, shape=(755, 850, 3)):
        mask = np.zeros(shape, dtype=np.uint8)
        mask_json_filename = filename.replace(
            'images', 'masks').replace('.PNG', '.json')
        with open(mask_json_filename, 'r') as f:
            mask_json = json.load(f)
        polygons = [shape['points'] for shape in mask_json['shapes']]
        for polygon in polygons:
            points = np.array(polygon, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], (255, 255, 255))

        return mask

    def __getitem__(self, index):
        image = self.get_image(self.img_filenames[index])
        mask = self.get_mask(self.img_filenames[index], image.shape)

        image = F.to_image(image)
        mask = F.to_image(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = F.resize(image, 400, antialias=None)
            mask = F.resize(mask, 400, antialias=None)

        if self.grayscale_mask:
            mask = F.to_grayscale(mask)

        image = F.to_dtype(image, torch.float32, scale=True)
        mask = F.to_dtype(mask, torch.float32, scale=True)

        return {'image': image, 'mask': mask}
