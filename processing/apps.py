from django.apps import AppConfig
from pathlib import Path
import torch

from segmentation.models.unet import UNet


class ProcessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'processing'

    checkpoint = torch.load(Path('segmentation/models/saved_models/unet_saved_v2.pth'))

    model = UNet(3, 1)
    model.load_state_dict(checkpoint)
    model.eval()
