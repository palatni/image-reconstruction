"""
This module contains a dataset that is used for image recovery.
"""

from typing import Optional, Tuple
import numpy as np
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    """
    The dataset used for image recovery task.
    """

    def __init__(
        self,
        img: np.ndarray,
        mask: None | np.ndarray = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            img (np.ndarray): Target image for the recovery task.
            mask (None, np.ndarray, optional): A mask labeling positions
                that are allowed to be sampled. If None is passed,
                all positions are allowed for sampling. Defaults to None.
        """
        self._img = img.astype(np.float32)
        self._spatial_scales = np.array(img.shape[:-1])
        if mask is None:
            mask = np.ones(self._img.shape[:-1], dtype=bool)
        self._positions = np.argwhere(mask)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples an element and its RGB value from internal position set.

        Args:
            index (int): An index for the pixel position.

        Returns:
            [np.ndarray, np.ndarray]: Position and RGB value.
        """
        pos = self._positions[index]
        return (
            (pos / self._spatial_scales).astype(np.float32),
            self._img[pos[0], pos[1]],
        )

    def __len__(self) -> int:
        """
        Returns the total number of pixel positions

        Returns:
            int: Total number of positions.
        """
        return self._positions.shape[0]
