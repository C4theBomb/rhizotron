import os
import logging
from pathlib import Path
import torch

from django.core.management.base import BaseCommand

from segmentation.models import UNet


class Command(BaseCommand):
    help = 'Export model weights to a pth file that can be loaded with torch.load().'

    def __init__(self):
        self.logger = logging.getLogger('main')

    def add_arguments(self, parser):
        parser.add_argument('target', type=str, default=None, help='Target checkpoint')
        parser.add_argument('--output', type=str, default='segmentation/models/saved_models', help='Output directory')

        parser.add_argument('--name', type=str, default='saved', help='Name of the model')
        parser.add_argument('--model', type=str, default='unet', choices=['unet'], help='Model to use')

    def handle(self, *args, **options):
        if not os.path.exists(options['output']):
            os.makedirs(options['output'])

        model = UNet(3, 1)

        checkpoint = torch.load(options['target'])
        model_weights = checkpoint['state_dict']

        for key in list(model_weights):
            model_weights[key.replace('model.', '')] = model_weights.pop(key)

        model.load_state_dict(model_weights)
        model.eval()
        
        output_path = Path(options['output'], '{}_{}'.format(options["model"], options["name"])).with_suffix('.pth')

        torch.save(model.state_dict(), output_path)
        self.logger.info(f'Saved model to {output_path}')
