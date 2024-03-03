from unittest import TestCase

from segmentation.models.model_types import ModelType
from segmentation.models import UNet, ResNet


class TestModelType(TestCase):
    def test_str(self):
        self.assertEqual(str(ModelType.UNET), "unet")
        self.assertEqual(str(ModelType.RESNET18), "resnet18")
        self.assertEqual(str(ModelType.RESNET34), "resnet34")
        self.assertEqual(str(ModelType.RESNET50), "resnet50")
        self.assertEqual(str(ModelType.RESNET101), "resnet101")
        self.assertEqual(str(ModelType.RESNET152), "resnet152")

    def test_get_model(self):
        unet_model = ModelType.UNET.get_model(3, 1)
        self.assertIsInstance(unet_model, UNet)

        resnet18_model = ModelType.RESNET18.get_model(3, 1)
        self.assertIsInstance(resnet18_model, ResNet)

        resnet34_model = ModelType.RESNET34.get_model(3, 1)
        self.assertIsInstance(resnet34_model, ResNet)

        resnet50_model = ModelType.RESNET50.get_model(3, 1)
        self.assertIsInstance(resnet50_model, ResNet)

        resnet101_model = ModelType.RESNET101.get_model(3, 1)
        self.assertIsInstance(resnet101_model, ResNet)

        resnet152_model = ModelType.RESNET152.get_model(3, 1)
        self.assertIsInstance(resnet152_model, ResNet)
