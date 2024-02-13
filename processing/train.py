from pathlib import Path
import argparse
import logging

import torch
from torch import Generator
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping, RichProgressBar, BatchSizeFinder
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import TensorBoardLogger

from data import LabelmeDataset, PRMIDataset
from models import TrainingModel, UNet


def log_function(func):
    def wrapper(*args, **kwargs):
        logging.info(f'Running function {func.__name__}')
        out = func(*args, **kwargs)
        logging.info(f'Finished running function {func.__name__}')

        return out

    return wrapper


class TrainingDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, num_workers, prefetch_factor):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)


@log_function
def load_danforth_dataset():
    dataset = LabelmeDataset('data/danforth/images', 'data/danforth/masks')
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [0.80, 0.15, 0.05], generator=Generator().manual_seed(0))

    val_dataset.dataset.transform = None
    test_dataset.dataset.transform = None

    return train_dataset, val_dataset, test_dataset


@log_function
def load_prmi_dataset():
    train_dataset = PRMIDataset('data/PRMI/train/images', 'data/PRMI/train/masks_pixel_gt')
    val_dataset = PRMIDataset('data/PRMI/val/images', 'data/PRMI/val/masks_pixel_gt')
    test_dataset = PRMIDataset('data/PRMI/test/images', 'data/PRMI/test/masks_pixel_gt')

    val_dataset.transform = None
    test_dataset.transform = None

    return train_dataset, val_dataset, test_dataset


@log_function
def train_model(trainer, model, data_module, ckpt_path=None):
    model.train()

    trainer.fit(model, data_module, ckpt_path=ckpt_path)


@log_function
def test_model(trainer, model, data_module, ckpt_path=None):
    model.eval()

    with torch.no_grad():
        trainer.test(model, data_module, ckpt_path=ckpt_path)


@log_function
def main(args):
    L.seed_everything(0)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('medium')
    logging.info(f'Using PyTorch version: {torch.__version__}')
    logging.info(f'Running with arguments: {vars(args)}')
    logging.info(f'Using device: {device}')

    if args.model == 'unet':
        model = TrainingModel(UNet, learning_rate=args.learning_rate)

    if args.dataset == 'danforth':
        train_dataset, val_dataset, test_dataset = load_danforth_dataset()
    elif args.dataset == 'prmi':
        train_dataset, val_dataset, test_dataset = load_prmi_dataset()

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{step}-{val_loss:0.2f}', dirpath=f'checkpoints/{args.model_dir}',
                                          monitor='val_loss', save_top_k=5, mode='min', save_last=True)

    trainer = L.Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        max_epochs=args.epochs,
        log_every_n_steps=1,
        precision='bf16-mixed',
        logger=TensorBoardLogger(save_dir=f'logs/{args.model_dir}'),
        profiler=SimpleProfiler(dirpath=f'logs/{args.model_dir}/profiler', filename='perf_logs'),
        callbacks=[
            RichProgressBar(),
            DeviceStatsMonitor(cpu_stats=True),
            EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
            checkpoint_callback,
        ]
    )

    data = TrainingDataModule(train_dataset, val_dataset, test_dataset, args.batch_size, args.num_workers,
                              args.prefetch_factor)

    if args.train:
        ckpt_path = 'last' if args.resume_last else None
        train_model(trainer, model, data, ckpt_path=ckpt_path)

    if args.test:
        ckpt_path = args.checkpoint if args.checkpoint else 'last'
        test_model(trainer, model, data, ckpt_path=ckpt_path)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler('logs/run.log'), logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        prog='train_model',
        description='Train a model to segment images of plant roots'
    )

    parser.add_argument('model_dir', type=str, help='Name of the model')

    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--model', type=str, default='unet', choices=['unet'], help='Model to use')
    parser.add_argument('--dataset', type=str, default='danforth', choices=['danforth', 'prmi'], help='Dataset to use')

    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--resume_last', action='store_true', default=False,
                        help='Resume training from last checkpoint')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    args = parser.parse_args()

    Path('checkpoints', args.model_dir).mkdir(parents=True, exist_ok=True)
    Path('logs', args.model_dir).mkdir(parents=True, exist_ok=True)

    try:
        main(args)
    except Exception as e:
        logging.exception(e)
    finally:
        logging.info('Finished running script')
