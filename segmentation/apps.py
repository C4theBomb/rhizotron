from django.apps import AppConfig

import torch

class SegmentationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'segmentation'

    model = torch.load('unet.pth')
    model.eval()
