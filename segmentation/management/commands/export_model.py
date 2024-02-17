import torch
import argparse
from pathlib import Path
import logging
import os
from django.core.management.base import BaseCommand

from models import UNet


class Command(BaseCommand):
    help = 'Save model as pth file'

    def __init__(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')

        self.logger = logging.getLogger('main')

    def add_arguments(self, parser):
        parser.add_argument('target', type=str, default=None, help='Target checkpoint')
        parser.add_argument('--output', type=str, default='models/saved_models', help='Output directory')

        parser.add_argument('--name', type=str, default='saved', help='Name of the model')
        parser.add_argument('--version', type=str, default='v1', help='Version of the model')
        parser.add_argument('--model', type=str, default='unet', choices=['unet'], help='Model to use')

    def handle(self, *args, **options):
        if os.path.exists(args.output):
            os.makedirs(args.output)

        model = UNet(3, 1)

        checkpoint = torch.load(args.target)
        model_weights = checkpoint['state_dict']

        for key in list(model_weights):
            model_weights[key.replace('model.', '')] = model_weights.pop(key)

        model.load_state_dict(model_weights)
        model.eval()

        torch.save(model.state_dict(), Path(args.output, f'{args.model}_{args.name}_{args.version}.pth'))
        self.logger.info(f'Saved model to {args.output}')
