import os
import cv2
import numpy as np
import pandas as pd

from pathlib import Path

import logging
import argparse

from utils import root_analysis, file_management


def main(args):
    image_filenames = file_management.get_image_filenames(args.target, args.recursive)

    measurements = pd.DataFrame(columns=['image', 'layer', 'root_count',
                                'total_root_length', 'total_root_area', 'root_diameter', 'total_root_volume'])

    try:
        for index, image_filename in enumerate(image_filenames):
            logging.info(f'Running image {index + 1} of {len(image_filenames)}: {image_filename}')

            tube_lower_end = int(os.path.basename(image_filename).split(
                '_')[2].removeprefix('L').removesuffix('.png').split('-')[0])
            tube_higher_end = int(os.path.basename(image_filename).split(
                '_')[2].removeprefix('L').removesuffix('.png').split('-')[1])
            segments = tube_higher_end - tube_lower_end + 1

            image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            segment_width = image.shape[1] // segments

            for layer in range(tube_lower_end, tube_higher_end + 1):
                segment = image[:, (layer - 1) * segment_width:layer * segment_width]
                metrics = root_analysis.calculate_metrics(segment, 0.2581)

                measurements.loc[len(measurements)] = {
                    'image': image_filename,
                    'layer': layer,
                    **metrics
                }

            logging.info(f'Completed image {index + 1} of {len(image_filenames)}: {image_filename}')
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt detected, saving progress')
    finally:
        measurements = measurements.round(4)
        measurements.to_csv(Path(args.output, 'layered_measurements.csv'), index=False)


if __name__ == '__main__':
    Path('logs').mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler('logs/run.log'), logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        prog='build_composites',
        description='Build composites from images'
    )

    parser.add_argument('--target', type=str, default=None, help='Target directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--recursive', action='store_true', help='Recursively search for images')
    parser.add_argument('--scaling_factor', type=float, default=0.2581, help='Scaling factor')

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
    finally:
        logging.info('Finished running script')
