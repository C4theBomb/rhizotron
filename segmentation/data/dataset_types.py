from enum import Enum

from torch.utils.data import Dataset

from .labelme import LabelmeDataset
from .prmi import PRMIDataset


class DatasetType(Enum):
    LABELME = 'labelme'
    PRMI = 'prmi'

    def __str__(self) -> str:
        return self.value

    def get_dataset(self, *args) -> Dataset:
        match self.value:
            case 'labelme':
                return LabelmeDataset(*args)
            case 'prmi':
                return PRMIDataset(*args)
            case _:
                raise ValueError(f'Invalid dataset type: {self.value}')
