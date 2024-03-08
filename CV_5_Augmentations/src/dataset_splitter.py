import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratify_shuffle_split_subsets(
    annotation: pd.DataFrame,
    train_fraction: float = 0.8,
    test_size: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбиение датасета на train/valid/test."""
    
    train_subset, val_test_subset = train_test_split(annotation, train_size=train_fraction, shuffle=True, stratify=annotation['label_num'])
    valid_subset, test_subset = train_test_split(val_test_subset, test_size=test_size, shuffle=True, stratify=val_test_subset['label_num'])
    logging.info('Stratifying dataset is completed.')

    return train_subset, valid_subset, test_subset
