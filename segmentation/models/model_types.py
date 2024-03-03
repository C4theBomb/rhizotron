from enum import Enum
from torch import nn

from .unet import UNet
from .resnet import ResNet


class ModelType(Enum):
    UNET = UNet
    RESNET18 = ResNet
    RESNET34 = ResNet
    RESNET50 = ResNet
    RESNET101 = ResNet
    RESNET152 = ResNet

    def __str__(self) -> str:
        return self.name.lower()

    def get_model(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> nn.Module:
        if self == ModelType.UNET:
            return self.value(in_channels, out_channels)
        else:
            if self in (ModelType.RESNET18, ModelType.RESNET34, ModelType.RESNET50):
                return self.value(in_channels, out_channels, num_layers=int(self.name[-2:]), dropout=dropout)
            elif self in (ModelType.RESNET101, ModelType.RESNET152):
                return self.value(in_channels, out_channels, num_layers=int(self.name[-3:]), dropout=dropout)
            else:
                raise ValueError(f'Invalid model type: {self}')

    @staticmethod
    def from_string(s: str) -> 'ModelType':
        match s.lower():
            case 'unet':
                return ModelType.UNET
            case 'resnet18':
                return ModelType.RESNET18
            case 'resnet34':
                return ModelType.RESNET34
            case 'resnet50':
                return ModelType.RESNET50
            case 'resnet101':
                return ModelType.RESNET101
            case 'resnet152':
                return ModelType.RESNET152
            case _:
                raise ValueError(f'Invalid model type: {s}')
