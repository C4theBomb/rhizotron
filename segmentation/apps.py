from django.apps import AppConfig
from pathlib import Path
import torch

from processing.models.unet import UNet


class SegmentationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'segmentation'

    checkpoint = torch.load(Path('processing/models/saved_models/unet_saved_v2.pth'))

    model = UNet(3, 1)
    model.load_state_dict(checkpoint)
    model.eval()
