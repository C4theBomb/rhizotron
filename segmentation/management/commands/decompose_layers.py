import os
import logging
import cv2
import pandas as pd

from django.core.management.base import BaseCommand, CommandParser

from segmentation.utils import root_analysis, file_management


class Command(BaseCommand):
    help = 'Decompose images layers and calculate root metrics for each layer.'

    def __init__(self):
        self.logger = logging.getLogger('main')

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('target', type=str, default=None, help='Target directory')
        parser.add_argument('--output', type=str, default='output', help='Output directory')
        parser.add_argument('--recursive', action='store_true', help='Recursively search for images')
        parser.add_argument('--scaling_factor', type=float, default=0.2581, help='Scaling factor')

    def handle(self, *args, **options) -> None:
        if not os.path.exists(options['output']):
            os.makedirs(options['output'])

        image_filenames = file_management.get_image_filenames(options['target'], options['recursive'])

        measurements = pd.DataFrame(
            columns=[
                'image',
                'layer',
                'root_count',
                'average_root_diameter',
                'total_root_length',
                'total_root_area',
                'total_root_volume'])

        try:
            for index, image_filename in enumerate(image_filenames):
                self.logger.info(f'Running image {index + 1} of {len(image_filenames)}: {image_filename}')

                tube_lower_end = int(os.path.basename(image_filename).split(
                    '_')[2].removeprefix('L').removesuffix('.png').split('-')[0])
                tube_higher_end = int(os.path.basename(image_filename).split(
                    '_')[2].removeprefix('L').removesuffix('.png').split('-')[1])
                segments = tube_higher_end - tube_lower_end + 1

                image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
                segment_width = image.shape[1] // segments

                for layer in range(tube_lower_end, tube_higher_end + 1):
                    segment = image[:, (layer - 1) * segment_width:layer * segment_width]
                    metrics = root_analysis.calculate_metrics(segment, options['scaling_factor'])

                    measurements.loc[len(measurements)] = {
                        'image': image_filename,
                        'layer': layer,
                        **metrics
                    }

                self.logger.info(f'Completed image {index + 1} of {len(image_filenames)}: {image_filename}')
        except KeyboardInterrupt:
            pass
        finally:
            measurements = measurements.round(4)
            measurements.to_csv(f'{options['output']}/layered_measurements.csv', index=False)
            self.logger.info(f'Saved measurements to {options["output"]}/layered_measurements.csv')
