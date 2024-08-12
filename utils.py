"""
This module contains helper functions for the scripts.

"""
from typing import Dict
from os import PathLike
from PIL import Image
import numpy as np
import yaml


def load_image(file_path: str | PathLike) -> np.ndarray:
    """
    A helper function that loads the image.

    Args:
        file_path (str | PathLike): Path to the file.

    Returns:
        np.ndarray: Loaded image.
    """
    img = Image.open(file_path)
    img.load()
    return np.asarray(img) / 255


def load_yaml(file_path: str | PathLike) -> Dict:
    """
    A helper function that reads config file

    Args:
        file_path (str | PathLike): Path to the file.

    Returns:
        Dict: Config dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
