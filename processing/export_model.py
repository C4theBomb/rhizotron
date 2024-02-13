import torch
import argparse
from pathlib import Path
import logging

from models import UNet


def main(args):
    model = UNet(3, 1)

    checkpoint = torch.load(args.target)
    model_weights = checkpoint['state_dict']

    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model.eval()

    torch.save(model.state_dict(), Path(args.output, f'{args.model}_{args.name}_{args.version}.pth'))


if __name__ == "__main__":
    Path('logs').mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler('logs/run.log'), logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        prog='export_model',
        description='Save model as pth file'
    )

    parser.add_argument('target', type=str, default=None, help='Target checkpoint')
    parser.add_argument('--output', type=str, default='models/saved_models', help='Output directory')

    parser.add_argument('--name', type=str, default='saved', help='Name of the model')
    parser.add_argument('--version', type=str, default='v1', help='Version of the model')
    parser.add_argument('--model', type=str, default='unet', choices=['unet'], help='Model to use')

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
    finally:
        logging.info('Finished running script')
