import os
import logging
import cv2
import numpy as np
import pandas as pd

from django.core.management.base import BaseCommand, CommandParser

from segmentation.utils import file_management


class Command(BaseCommand):
    help = 'Combine images of individual layers into composite images.'

    def __init__(self):
        self.logger = logging.getLogger('main')

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('target', type=str, help='Target directory')
        parser.add_argument('--output', type=str, default='output/concat', help='Output directory')
        parser.add_argument('--recursive', action='store_true', help='Recursively search for images')

    def handle(self, *args, **options) -> None:
        image_filepaths = file_management.get_image_filenames(options['target'], options['recursive'])

        df = pd.DataFrame([], columns=['Date', 'DPI', 'Plant Type', 'Tube', 'Level', 'Path'])

        for index, filepath in enumerate(image_filepaths):
            filename_parts = filepath.split('/')
            date = filename_parts[-2].split('_')[0]
            dpi = filename_parts[-2].split('_')[2].removesuffix('dpi')
            plant_type = filename_parts[-1].split('_')[0]
            tube = filename_parts[-1].split('_')[1].removeprefix('T')
            level = filename_parts[-1].split('_')[2].removeprefix('L').removesuffix('.PNG')
            df.loc[index] = ([date, dpi, plant_type, tube, level, filepath])

        df['Date'] = pd.to_datetime(df['Date'], format='%m%d%Y')
        df['DPI'] = df['DPI'].astype(int)
        df['Tube'] = df['Tube'].astype(int)
        df['Level'] = df['Level'].astype(int)

        df.sort_values(by=['Date', 'Tube', 'Level'], inplace=True)

        groups = df.groupby(['Date', 'Tube', 'Plant Type'])

        for group_index, (name, group) in enumerate(groups):
            self.logger.info(
                f'Running image {group_index + 1} of {len(groups)}: {name[0].strftime("%m%d%Y")}, Tube {name[1]}')

            images = [cv2.imread(row['Path']) for _, row in group.iterrows()]
            min_level = group['Level'].min()
            max_level = group['Level'].max()
            for index in range(0, len(images) - 1):
                if group['DPI'].mean() == 100:
                    images[index] = images[index][:, :-50, :]
                elif group['DPI'].mean() == 300:
                    images[index] = images[index][:, :, :]

            image = np.concatenate(images, axis=1)

            if not os.path.exists(f'{options["output"]}/{name[0].strftime("%m%d%Y")}'):
                os.makedirs(f'{options["output"]}/{name[0].strftime("%m%d%Y")}')
            cv2.imwrite(
                f'{options["output"]}/{name[0].strftime("%m%d%Y")}/{name[2]}_T{name[1]}_L{min_level}-{max_level}.png',
                image)

            self.logger.info(
                f'Completed image {group_index + 1} of {len(groups)}: {name[0].strftime("%m%d%Y")}, Tube {name[1]}')
