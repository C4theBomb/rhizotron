import logging
import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
import torch

from django.core.management.base import BaseCommand, CommandParser

from segmentation.models import TrainingModel, ModelType
from segmentation.data import TrainingDataModule, DatasetType


class Command(BaseCommand):
    def __init__(self):
        self.loggger = logging.getLogger(__name__)

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('model_name', type=str, help='Name of the model')
        parser.add_argument('dataset_dir', type=str,
                            help='Directory of the dataset')

        parser.add_argument('--train', action='store_true',
                            help='Train the model')
        parser.add_argument('--test', action='store_true',
                            help='Test the model')
        parser.add_argument('--model', type=ModelType.from_string, default=ModelType.UNET,
                            choices=list(ModelType), help='Model to use')
        parser.add_argument('--dataset_type', type=DatasetType.from_string, default=DatasetType.LABELME,
                            choices=list(DatasetType), help='Type of dataset')

        parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--min_zoom', type=float, default=0.5, help='Minimum zoom for random resized crop')
        parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
        parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
        parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
        parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
        parser.add_argument('--resume_last', action='store_true', default=False, help='Resume training from last checkpoint')

        parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    def handle(self, *args, **options) -> None:
        L.seed_everything(0)
        device = torch.device(
            "cuda" if options['cuda'] and torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision('medium')
        logging.info(f'Using PyTorch version: {torch.__version__}')
        logging.info(f'Running with arguments: {options}')
        logging.info(f'Using device: {device}')

        logging.info(f'Using model: {options["model"].value.__name__}')
        model = TrainingModel(
            options['model'], learning_rate=options['learning_rate'], dropout=options['dropout'])

        logging.info(
            f'Using dataset: {options["dataset_type"].value.__name__}')
        data_module = TrainingDataModule(options['dataset_dir'], options['dataset_type'],
                                         options['batch_size'], options['num_workers'], options['prefetch_factor'])

        checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{step}-{val_loss:0.2f}', dirpath=f'segmentation/checkpoints/{options["model_name"]}', monitor='val_loss', save_top_k=5, mode='min', save_last=True)

        trainer = L.Trainer(
            accelerator='gpu' if device.type == 'cuda' else 'cpu',
            max_epochs=options['epochs'],
            log_every_n_steps=1,
            precision='bf16-mixed' if device.type == 'cuda' else None,
            logger=TensorBoardLogger(save_dir=f'logs/{options["model_name"]}'),
            profiler=SimpleProfiler(
                dirpath=f'logs/{options["model_name"]}/profiler', filename='perf_logs'),
            callbacks=[
                RichProgressBar(),
                DeviceStatsMonitor(cpu_stats=True),
                EarlyStopping(monitor='val_loss',
                              patience=options['patience'], mode='min'),
                checkpoint_callback,
            ]
        )

        logging.info('Starting training')
        if options['train']:
            model.train()

            ckpt_path = 'last' if options['resume_last'] else None
            trainer.fit(model, data_module, ckpt_path=ckpt_path)

        logging.info('Starting testing')
        if options['test']:
            model.eval()

            ckpt_path = options['checkpoint'] if options['checkpoint'] else 'last'
            with torch.no_grad():
                trainer.test(model, data_module, ckpt_path=ckpt_path)

        logging.info('Done')
