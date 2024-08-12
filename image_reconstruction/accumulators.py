"""
This module contains accumulators that gather
metrics and data during a training process.
"""
from typing import Tuple
import numpy as np


class MSEMetricsAccumulator:
    """
    An accumulator that gathers the cumulative sum
    and number of total samples, so that the MSE and PSNR
    metrics can be restored after the epoch end.
    """

    def __init__(self) -> None:
        """
        Initializes accumulator with zero sum and number of samples.
        """
        self._cum_square_error = 0
        self._num_samples = 0

    def update(self, new_mse: float, num_new_samples: int) -> None:
        """
        Updates the state of the accumulator.

        Args:
            new_mse (float): MSE loss gathered from the network.
            num_new_samples (int): Number of samples in the batch.
        """
        self._cum_square_error += new_mse * num_new_samples
        self._num_samples += num_new_samples

    def reset(self) -> None:
        """
        Resets the state of the accumulator with zeros.
        """
        self._cum_square_error = 0
        self._num_samples = 0

    @property
    def mse(self) -> float:
        """
        A property recovering MSE from the state of the accumulator.

        Returns:
            float: MSE value
        """
        return self._cum_square_error / self._num_samples

    @property
    def psnr(self) -> float:
        """
        A property recovering PSNR metric from the state of the accumulator.

        Returns:
            float: PSNR value
        """
        return -10 * np.log10(self.mse)


class ImgDataAccumulator:
    """
    An accumulator that gathers the recovered image pixels from the model
    during the training process from both training and validation subsets.
    """

    def __init__(self, img_shape: Tuple[int, int, int]) -> None:
        """
        Initializes the accumulator with a blank image.

        Args:
            img_shape (Tuple[int, int, int]): The shape of the target image.
        """
        self._img = np.zeros(img_shape)

    def update(self, scaled_pos: np.ndarray, channel_data: np.ndarray) -> None:
        """
        Updates the state of the accumulator

        Args:
            scaled_pos (np.ndarray): An array of stacked (h, w)
                positions of pixels.
            channel_data (np.ndarray): An array of stacked (r, g, b)
                values of pixels
        """
        h_max, w_max = self._img.shape[:-1]
        h_array = (scaled_pos[:, 0] * h_max).astype(np.int32)
        w_array = (scaled_pos[:, 1] * w_max).astype(np.int32)
        self._img[h_array, w_array] = channel_data

    def reset(self) -> None:
        """
        Fills the image with zeros.
        """
        self._img[:] = 0

    @property
    def img(self) -> np.ndarray:
        """
        A property returning the current predicted image

        Returns:
            np.ndarray: Predicted image.
        """
        return self._img
