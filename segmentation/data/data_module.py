from enum import Enum

from torch import Generator
from torch.utils.data import DataLoader, random_split
import lightning as L

from .labelme import LabelmeDataset
from .prmi import PRMIDataset


class DatasetType(Enum):
    LABELME = LabelmeDataset
    PRMI = PRMIDataset
    
    def __str__(self) -> str:
        return self.value.__name__.lower().replace('dataset', '')

    @staticmethod
    def from_string(s: str) -> 'DatasetType':
        match s.lower():
            case 'labelme':
                return DatasetType.LABELME
            case 'prmi':
                return DatasetType.PRMI
            case _:
                raise ValueError(f'Invalid dataset type: {s}')

class TrainingDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir: str, dataset_type: DatasetType, batch_size: int, num_workers: int, prefetch_factor: int) -> None:
        match dataset_type:
            case DatasetType.LABELME:
                dataset = dataset_type.value(dataset_dir)
                train_dataset, val_dataset, test_dataset = random_split(dataset, [0.80, 0.15, 0.05], generator=Generator().manual_seed(0))
            case DatasetType.PRMI:
                train_dataset = dataset_type.value(f'{dataset_dir}/train/images', f'{dataset_dir}/train/masks_pixel_gt')
                val_dataset = dataset_type.value(f'{dataset_dir}/val/images', f'{dataset_dir}/val/masks_pixel_gt')
                test_dataset = dataset_type.value(f'{dataset_dir}/test/images', f'{dataset_dir}/test/masks_pixel_gt')
                
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)