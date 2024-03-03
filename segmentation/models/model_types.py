from enum import Enum
from torch import nn

from .unet import UNet
from .resnet import ResNet


class ModelType(Enum):
    UNET = 'unet'
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'

    def __str__(self) -> str:
        return self.value

    def get_model(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> nn.Module:
        if self == ModelType.UNET:
            return UNet(in_channels, out_channels)
        else:
            if self in (ModelType.RESNET18, ModelType.RESNET34, ModelType.RESNET50):
                return ResNet(in_channels, out_channels, num_layers=int(self.name[-2:]), dropout=dropout)
            elif self in (ModelType.RESNET101, ModelType.RESNET152):
                return ResNet(in_channels, out_channels, num_layers=int(self.name[-3:]), dropout=dropout)
            else:
                raise ValueError(f'Invalid model type: {self}')
