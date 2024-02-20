from PIL import Image as PILImage
import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as F

from .masks import threshold


def predict(model: nn.Module, image_path: str, area_threshold: int = 15) -> PILImage.Image:
    """
    Predicts the segmentation mask for an input image using a given model.

    Args:
        model (nn.Module): The segmentation model.
        image_path (str): The path to the input image.
        area_threshold (int, optional): The threshold for filtering small regions in the segmentation mask. 
            Defaults to 15.

    Returns:
        PIL.Image.Image: The predicted segmentation mask as a PIL image.
    """
    image = PILImage.open(image_path)
    image = np.array(image)
    image = image[:, :, :3]
    image = F.to_image(image)
    image = F.to_dtype(image, torch.float32, scale=True)

    image = image.unsqueeze(0)
    image = model(image).detach()
    image = image.squeeze(0, 1)
    image = image.numpy().astype(np.uint8)
    image = threshold(image, area_threshold)
    image = F.to_pil_image(image)

    return image
