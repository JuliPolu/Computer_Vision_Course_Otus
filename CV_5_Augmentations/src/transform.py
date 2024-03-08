from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from torch import Tensor


TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    aug_yaml: str = None,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        if aug_yaml:
            transforms.extend(
                albu.load(aug_yaml, data_format='yaml')
            )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)


def cv_image_to_tensor(img: NDArray, normalize: bool = True) -> Tensor:
    ops = [
        ToTensorV2(),
    ]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)['image']


def denormalize(
    img: NDArray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
):
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)['image'] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray:
    return tensor.permute(1, 2, 0).cpu().numpy()

