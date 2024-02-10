from PIL import Image
import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from . import threshold


def predict(model, image_path, area_threshold=0.5):
    image = Image.open(image_path)
    image = np.array(image)
    image = image[:, :, :3]
    image = F.to_image(image)
    image = F.to_dtype(image, torch.float32, scale=True)

    image = image.unsqueeze(0)
    image = model(image).detach()
    image = image.squeeze(0, 1)
    image = image.numpy().astype(np.uint8)
    image = threshold.threshold_mask(image, area_threshold)
    image = F.to_pil_image(image)

    return image
