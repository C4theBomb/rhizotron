from unittest import TestCase
import torch

from segmentation.models.resnet import ResNet


class ResNetTest(TestCase):
    def test_resnet18(self):
        model = ResNet(3, 1, 18)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_resnet34(self):
        model = ResNet(3, 1, 34)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_resnet50(self):
        model = ResNet(3, 1, 50)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_resnet101(self):
        model = ResNet(3, 1, 101)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_resnet152(self):
        model = ResNet(3, 1, 152)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, (1, 1, 256, 256))
