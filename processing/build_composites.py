import os
from pathlib import Path

import argparse
import logging

import numpy as np
import pandas as pd
import cv2

from utils import file_management


def main(args):
    image_filepaths = file_management.get_image_filenames(args.target, args.recursive)

    df = pd.DataFrame([], columns=['Date', 'DPI', 'Tube', 'Level', 'Path'])

    for index, filepath in enumerate(image_filepaths):
        filename_parts = filepath.split('/')
        date = filename_parts[-2].split('_')[0]
        dpi = filename_parts[-2].split('_')[2].removesuffix('dpi')
        tube = filename_parts[-1].split('_')[1].removeprefix('T')
        level = filename_parts[-1].split('_')[2].removeprefix('L').removesuffix('.PNG')
        df.loc[index] = ([date, dpi, tube, level, filepath])

    df['Date'] = pd.to_datetime(df['Date'], format='%m%d%Y')
    df['DPI'] = df['DPI'].astype(int)
    df['Tube'] = df['Tube'].astype(int)
    df['Level'] = df['Level'].astype(int)

    df.sort_values(by=['Date', 'Tube', 'Level'], inplace=True)

    groups = df.groupby(['Date', 'Tube'])

    for group_index, (name, group) in enumerate(groups):
        logging.info(f'Running image {group_index + 1} of {len(groups)}: {name[0].strftime("%m%d%Y")}, Tube {name[1]}')

        images = [cv2.imread(row['Path']) for _, row in group.iterrows()]
        min_level = group['Level'].min()
        max_level = group['Level'].max()
        for index in range(0, len(images) - 1):
            if group['DPI'].mean() == 100:
                images[index] = images[index][:, :-50, :]
            elif group['DPI'].mean() == 300:
                images[index] = images[index][:, :, :]

        image = np.concatenate(images, axis=1)

        if args.output == 'output':
            if not os.path.exists(f'./output/concatenated/{name[0].strftime("%m%d%Y")}'):
                os.makedirs(f'./output/concatenated/{name[0].strftime("%m%d%Y")}')
            cv2.imwrite(
                f'./output/concatenated/{name[0].strftime("%m%d%Y")}/CS_T{name[1]}_L{min_level}-{max_level}.png', image)
        else:
            if not os.path.exists(f'{args.output}/{name[0].strftime("%m%d%Y")}'):
                os.makedirs(f'{args.output}/{name[0].strftime("%m%d%Y")}')
            cv2.imwrite(f'{args.output}/{name[0].strftime("%m%d%Y")}/CS_T{name[1]}_L{min_level}-{max_level}.png', image)

        logging.info(
            f'Completed image {group_index + 1} of {len(groups)}: {name[0].strftime("%m%d%Y")}, Tube {name[1]}')


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

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
    finally:
        logging.info('Finished running script')
