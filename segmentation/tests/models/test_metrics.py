from unittest import TestCase

import torch

from segmentation.models.metrics import Accuracy, Dice


class AccuracyTest(TestCase):
    def test_accuracy(self):
        accuracy = Accuracy()
        y_true = torch.tensor([0, 1, 1, 0])
        y_pred = torch.tensor([0, 1, 0, 1])
        result = accuracy(y_pred, y_true)
        self.assertEqual(result, 0.5)


class DiceTest(TestCase):
    def test_dice(self):
        dice = Dice()
        y_true = torch.tensor([0, 1, 1, 0])
        y_pred = torch.tensor([0, 1, 0, 1])
        result = dice(y_pred, y_true)
        self.assertAlmostEqual(result.item(), 0.5, 4)
