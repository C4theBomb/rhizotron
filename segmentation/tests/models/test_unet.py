from unittest import TestCase
import torch

from segmentation.models.unet import UNet


class UNetTest(TestCase):
    def test_unet(self):
        model = UNet(3, 1)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))
