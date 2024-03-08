from pathlib import Path
from typing import Optional, Union

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class ButterflyDataset(Dataset):
    def __init__(
        self,
        df: np.ndarray,
        image_folder: Path,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.df = df
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx: int):
        row = self.df[idx]

        image_path = self.image_folder / row[0]
        label = int(row[2])

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {'image': image, 'label': label}

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['label']

    def __len__(self) -> int:
        return len(self.df)
