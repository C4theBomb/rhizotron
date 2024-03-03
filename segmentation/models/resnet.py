from typing import Type
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.v2 import functional as F_image


class ResnetBlock(nn.Module):
    def __init__(
            self,
            num_layers: int,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: nn.Module = None):
        super().__init__()

        self.num_layers = num_layers
        self.downsample = downsample

        if self.num_layers <= 34:
            self.step = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.step = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.step(x)
        x += identity

        return F.relu(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.step(x)
        return x


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            num_layers: int = 18,
            dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers

        match self.num_layers:
            case 18:
                layers = [2, 2, 2, 2]
                self.expansion = 1
            case 34:
                layers = [3, 4, 6, 3]
                self.expansion = 1
            case 50:
                layers = [3, 4, 6, 3]
                self.expansion = 4
            case 101:
                layers = [3, 4, 23, 3]
                self.expansion = 4
            case 152:
                layers = [3, 8, 36, 3]
                self.expansion = 4

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.up_step1 = nn.ConvTranspose2d(512 * self.expansion, 512, kernel_size=2, stride=2)
        self.up_step2 = CNNBlock(512 + 256 * self.expansion, 256)
        self.up_step3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up_step4 = CNNBlock(256 + 128 * self.expansion, 128)
        self.up_step5 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up_step6 = CNNBlock(128 + 64 * self.expansion, 64)
        self.up_step7 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_step8 = CNNBlock(128, out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def _make_layer(
            self,
            out_channels: int,
            blocks: int,
            stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        layers = []
        layers.append(ResnetBlock(self.num_layers, self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * self.expansion

        for _ in range(1, blocks):
            layers.append(ResnetBlock(self.num_layers, out_channels * self.expansion, out_channels))

        layers = nn.Sequential(*layers)

        return layers

    def _pad_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height = int(32 * round(x.shape[2] / 32))
        width = int(32 * round(x.shape[3] / 32))

        resized = F_image.resize(x, (height, width), antialias=None)

        return resized, (x.shape[2], x.shape[3])

    def _unpad_output(self, x: torch.Tensor, original_dims: tuple[int, int]) -> torch.Tensor:
        return F_image.resize(x, original_dims, antialias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_dims = self._pad_input(x)

        x1 = self.conv1(x)
        x1 = self.dropout(x1)

        x2 = self.pool(x1)
        x2 = self.layer1(x2)
        x2 = self.dropout(x2)

        x3 = self.layer2(x2)
        x3 = self.dropout(x3)

        x4 = self.layer3(x3)
        x4 = self.dropout(x4)

        encoder_output = checkpoint(self.layer4, x4, use_reentrant=False)

        y4 = self.up_step1(encoder_output)
        y4 = torch.cat([x4, y4], dim=1)
        y4 = self.dropout(y4)
        y4 = self.up_step2(y4)

        y3 = self.up_step3(y4)
        y3 = torch.cat([x3, y3], dim=1)
        y3 = self.dropout(y3)
        y3 = self.up_step4(y3)

        y2 = self.up_step5(y3)
        y2 = torch.cat([x2, y2], dim=1)
        y2 = self.dropout(y2)
        y2 = self.up_step6(y2)

        y1 = self.up_step7(y2)
        y1 = torch.cat([x1, y1], dim=1)
        y1 = self.dropout(y1)
        y1 = self.up_step8(y1)

        output = F.sigmoid(y1)

        output = self._unpad_output(output, original_dims)

        return output
