from pathlib import Path
import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

import torch
from torchvision.transforms.v2 import functional as F

from segmentation.models import ModelType

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from segmentation.utils import masks, file_management, root_analysis

from django.core.management.base import BaseCommand, CommandParser


class Command(BaseCommand):
    def __init__(self):
        self.logger = logging.getLogger('main')

    def get_image(self, filename: str, size: int = None) -> torch.Tensor:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = F.to_image(image)
        image = F.crop(image, 0, 0, image.shape[1] - 2, image.shape[2])

        if size is not None:
            image = F.resize(image, size, antialias=None)

        image = F.to_dtype(image, torch.float32, scale=True)

        return image

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('target', type=str, default=None, help='Target directory')
        parser.add_argument('--output', type=str, default='output', help='Output directory')
        parser.add_argument('--recursive', action='store_true', help='Recursively search for images')

        parser.add_argument(
            '--model',
            type=ModelType,
            default=ModelType.UNET,
            choices=list(ModelType),
            help='Model to use')
        parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')

        parser.add_argument('--save_mask', action='store_true', help='Save masks')
        parser.add_argument('--save_comparison', action='store_true', help='Compare images and masks')
        parser.add_argument('--save_labelme', action='store_true', help='Save masks in labelme format')

        parser.add_argument('--size', type=int, default=None, help='Size to resize images to')
        parser.add_argument('--scaling_factor', type=float, default=0.2581, help='Scaling factor for the images')
        parser.add_argument('--threshold_area', type=int, default=15, help='Threshold area for the mask')

        parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    def handle(self, *args, **options) -> None:
        if not os.path.exists(options['output']):
            os.makedirs(options['output'])

        device = torch.device(
            "cuda" if options['cuda'] and torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision('medium')
        logging.info(f'Using PyTorch version: {torch.__version__}')
        logging.info(f'Running with arguments: {options}')
        logging.info(f'Using device: {device}')

        model = options['model'].get_model(3, 1)

        checkpoint = torch.load(options['checkpoint'])

        model_weights = checkpoint['state_dict']

        for key in list(model_weights):
            model_weights[key.replace('model.', '')] = model_weights.pop(key)

        model.load_state_dict(model_weights)
        model.eval()
        model.to(device)

        image_filenames = file_management.get_image_filenames(options['target'], options['recursive'])

        measurements = pd.DataFrame(
            columns=[
                'image',
                'root_count',
                'average_root_diameter',
                'total_root_length',
                'total_root_area',
                'total_root_volume'])

        try:
            for index, image_filename in enumerate(image_filenames):
                logging.info(
                    f'Running image {index + 1} of {len(image_filenames)}: {image_filename}')

                original_image = self.get_image(image_filename, options['size'])

                with torch.no_grad():
                    image = torch.clone(original_image).to(device)
                    image = image.unsqueeze(0)
                    output = model(image)
                    output = output.squeeze(0, 1)
                    output = (output > 0.5).float()

                output = output.type(torch.uint8) * 255
                mask = output.cpu().numpy()

                if options['threshold_area'] > 0:
                    mask = masks.threshold(mask, options['threshold_area'])

                if options['save_mask']:
                    Path(os.path.join(options['output'], 'mask', os.path.relpath(
                        os.path.dirname(image_filename), options['target']))).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(os.path.join(options['output'], 'mask', os.path.relpath(
                        os.path.dirname(image_filename), options['target']), os.path.basename(image_filename)), mask)

                if options['save_comparison']:
                    figure = plt.figure(figsize=(10, 10))

                    figure.add_subplot(2, 1, 1)
                    plt.title('Image')
                    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu())
                    figure.add_subplot(2, 1, 2)
                    plt.title('Mask')
                    plt.imshow(mask, cmap='gray')

                    Path(os.path.join(options['output'], 'compare', os.path.relpath(
                        os.path.dirname(image_filename), options['target']))).mkdir(parents=True, exist_ok=True)
                    plt.savefig(os.path.join(options['output'], 'compare', os.path.relpath(
                        os.path.dirname(image_filename), options['target']), os.path.basename(image_filename)))

                    plt.close(figure)

                if options['save_labelme']:
                    Path(os.path.join(options['output'], 'labelme', os.path.relpath(
                        os.path.dirname(image_filename), options['target']))).mkdir(parents=True, exist_ok=True)

                    original_image = original_image.numpy().transpose((1, 2, 0)) * 255
                    original_image = original_image.astype(np.uint8)
                    original_image = cv2.cvtColor(
                        original_image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(
                        os.path.join(
                            options['output'],
                            'labelme',
                            os.path.relpath(
                                os.path.dirname(image_filename),
                                options['target']),
                            os.path.basename(image_filename)),
                        original_image)

                    labelme_json = masks.to_labelme(
                        os.path.basename(image_filename), mask)

                    with open(os.path.join(options['output'], 'labelme', os.path.relpath(
                            os.path.dirname(image_filename), options['target']), os.path.basename(image_filename).upper().replace('.PNG', '.json')), 'w') as f:
                        f.write(labelme_json)

                measurements.loc[index] = {'image': image_filename,
                                           **root_analysis.calculate_metrics(mask, options['scaling_factor'])}

                logging.info(
                    f'Completed image {index + 1} of {len(image_filenames)}: {image_filename}')
        except KeyboardInterrupt:
            pass
        finally:
            measurements = measurements.round(4)
            measurements.to_csv(os.path.join(options['output'], 'measurements.csv'), index=False)
            self.logger.info(f'Saved measurements to {options["output"]}/measurements.csv')
