"""
This module contains Classes that are intended to store
any data during the training process.
"""

from itertools import count
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, List
from os import PathLike
from abc import ABC, abstractmethod
import io
import csv
import av
import numpy as np
import torch
from streamlit.delta_generator import DeltaGenerator
from PIL import Image


_LOG_DOCSTRING = """
    This method is called each epoch in order to update the logger's state.

    Args:
        log_data (LoggingData): And instance of LoggingData that contains
            the current epoch info.
    """


def _get_init_docstring(
        file_path_default: None | str | PathLike = None,
        write_frequency_default: int = 10
) -> str:
    return f"""
        All loggers are intended to be initialized with target file path and
        the frequency of writing the data.

        Args:
            file_path (None | str | PathLike, optional): A path to the
                target file. Defaults to {file_path_default}.
            write_frequency (int, optional): frequency of writing.
                For instance, writing_frequency=10 means that the data will be
                stored once per 10 epochs during the training.
                Defaults to {write_frequency_default}.
        """


def _set_docstring(doc: str, prepend: bool = False) -> Callable:
    def decorator(func: Callable) -> Callable:
        if func.__doc__:
            if prepend:
                func.__doc__ = doc + func.__doc__
            else:
                func.__doc__ += doc
        else:
            func.__doc__ = doc
        return func
    return decorator


@dataclass
class LoggingData:
    """
    A dataclass that specifies the supported values that can be
    stored or which derivative data can be stored.
    """

    epoch_id: None | int = None
    train_mse: None | float = None
    val_mse: None | float = None
    train_psnr: None | float = None
    val_psnr: None | float = None
    pred_img: None | np.ndarray = None
    model: None | np.ndarray = None


class Logger(ABC):
    """
    A base class for logging.
    """

    @_set_docstring(_get_init_docstring())
    def __init__(
        self,
        file_path: None | str | PathLike = None,
        write_frequency: int = 10
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._write_frequency = write_frequency
        self._log_counter = count(1)

    @abstractmethod
    @_set_docstring(_LOG_DOCSTRING)
    def log(self, log_data: LoggingData) -> None:
        raise NotImplementedError()


class CSVLogger(Logger):
    """
    A logger that saves and updates a csv file that
    contains the next columns:
    |Eoch ID | Train MSE | Val MSE | Train PSRR | Val PSNR|
    """

    @_set_docstring(_get_init_docstring('metric_log.csv'))
    def __init__(
        self,
        file_path: str | PathLike = 'metric_log.csv',
        write_frequency: int = 10
    ) -> None:
        super().__init__(file_path=file_path, write_frequency=write_frequency)

        with open(self._file_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                ('Epoch ID', 'Train MSE', 'Val MSE', 'Train PSNR', 'Val PSNR')
            )

        self._cached_data = []

    @_set_docstring(_LOG_DOCSTRING)
    def log(self, log_data: LoggingData) -> None:
        self._cached_data.append(
            [
                log_data.epoch_id,
                log_data.train_mse,
                log_data.val_mse,
                log_data.train_psnr,
                log_data.val_psnr,
            ]
        )

        if not next(self._log_counter) % self._write_frequency:
            with open(self._file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self._cached_data)
            self._cached_data = []


class VideoLogger(Logger):
    """
    A logger that stores a frame into a buffer each
    write_frequency epochs. After the training process,
    a video that pictures the prediction evolution
    during the training can be stored.
    """

    @_set_docstring(_get_init_docstring('train_video.mp4', 1), prepend=True)
    def __init__(
        self,
        file_path: str | PathLike = 'train_video.mp4',
        write_frequency: int = 1,
        fps: int = 24,
    ) -> None:
        """
            fps (int, optional): fps. Defaults to 24.
        """
        super().__init__(file_path=file_path, write_frequency=write_frequency)
        self._output_memory_file = io.BytesIO()
        self._output = av.open(self._output_memory_file, 'w', format='mp4')
        self._stream = self._output.add_stream('h264', str(fps))
        self._stream.pix_fmt = 'yuv444p'
        self._stream.options = {'crf': '17'}
        self._stream_shape_initialized = False

    @_set_docstring(_LOG_DOCSTRING)
    def log(self, log_data: LoggingData) -> None:
        if not self._stream_shape_initialized:
            self._stream.height = log_data.pred_img.shape[0]
            self._stream.width = log_data.pred_img.shape[1]
            self._stream_shape_initialized = True
        log_id = next(self._log_counter)
        if not log_id % self._write_frequency:
            frame = av.VideoFrame.from_ndarray(
                (log_data.pred_img * 255).astype(np.uint8),
                format='rgb24',
            )
            packet = self._stream.encode(frame)
            self._output.mux(packet)

    def save_video(self) -> None:
        """
        Saves the video and closes the buffer.
        """
        packet = self._stream.encode(None)
        self._output.mux(packet)
        self._output.close()
        with open(self._file_path, 'wb') as f:
            f.write(self._output_memory_file.getbuffer())


class StateDictLogger(Logger):
    """
    A logger that stores the model's state dict.
    """

    @_set_docstring(_get_init_docstring('state_dict.pt'))
    def __init__(
        self,
        file_path: str | PathLike = 'state_dict.pt',
        write_frequency: int = 10
    ) -> None:
        super().__init__(file_path=file_path, write_frequency=write_frequency)

    @_set_docstring(_LOG_DOCSTRING)
    def log(self, log_data: LoggingData) -> None:
        if not next(self._log_counter) % self._write_frequency:
            torch.save(log_data.model.state_dict(), self._file_path)


class CurrentImgLogger(Logger):
    """
    A logger that stores and updates a predicted image during the training
    """

    @_set_docstring(_get_init_docstring('current_img.jpg'))
    def __init__(self, file_path: str | PathLike = 'current_img.jpg', write_frequency: int = 10) -> None:
        super().__init__(file_path, write_frequency)

    @_set_docstring(_LOG_DOCSTRING)
    def log(self, log_data: LoggingData) -> None:
        if not next(self._log_counter) % self._write_frequency:
            img = Image.fromarray((log_data.pred_img * 255).astype(np.uint8))
            img.save(self._file_path)


class StreamlitMetricLogger(Logger):
    """
    A logger that communicates with Streamlit framework in order
    to update its plots. Unfortunately the training process is
    too slow and laggy if the training is initiated in the same
    process as the Streamlit code.
    """
    @_set_docstring(_get_init_docstring('state_dict.pt'), prepend=True)
    def __init__(
        self,
        mse_line_chart: DeltaGenerator,
        psnr_line_chart: DeltaGenerator,
        write_frequency: int = 10
    ) -> None:
        """
            mse_line_chart (DeltaGenerator): a line chart for MSE plotting
            psnr_line_chart (DeltaGenerator): a line chart for PSNR plotting
        """
        super().__init__(write_frequency=write_frequency)
        self._mse_chart = mse_line_chart
        self._psnr_chart = psnr_line_chart
        self._mse_data = self._reset_data()
        self._psnr_data = self._reset_data()

    @staticmethod
    def _reset_data() -> Dict:
        """
        a method to reset data for plotting

        Returns:
            Dict: a dictionary with Training and Validation data
        """
        return {
            'Train': [],
            'Val': [],
        }

    @_set_docstring(_LOG_DOCSTRING)
    def log(self, log_data: LoggingData) -> None:
        self._mse_data['Train'].append(log_data.train_mse)
        self._mse_data['Val'].append(log_data.val_mse)
        self._psnr_data['Train'].append(log_data.train_psnr)
        self._psnr_data['Val'].append(log_data.val_psnr)

        log_id = next(self._log_counter)

        if log_id == 0:
            self._mse_chart = self._mse_chart.line_chart(
                self._mse_data, x_label='Epoch ID', y_label='MSE')
            self._psnr_chart = self._psnr_chart.line_chart(
                self._psnr_data, x_label='Epoch ID', y_label='PSNR')
            self._mse_data = self._reset_data()
            self._psnr_data = self._reset_data()
        elif not next(self._log_counter) % self._write_frequency:
            self._mse_chart.add_rows(self._mse_data)
            self._psnr_chart.add_rows(self._psnr_data)
            self._mse_data = self._reset_data()
            self._psnr_data = self._reset_data()
