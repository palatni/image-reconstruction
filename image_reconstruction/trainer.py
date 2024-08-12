"""
This module contains a trainer class that wraps all the
training routine
"""

from typing import Tuple, Iterable
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from image_reconstruction.accumulators import (
    MSEMetricsAccumulator,
    ImgDataAccumulator,
)
from image_reconstruction.dataset import ImgDataset
from image_reconstruction.loggers import Logger, LoggingData


class ReconstructionTrainer:
    def __init__(self,
                 img: np.ndarray,
                 model: nn.Module,
                 optimizer_type: Optimizer = Adam,
                 lr: float = 1e-3,
                 batch_size: int = 4096,
                 dataloader_workers: int = 4,
                 device: None | str = None,
                 loggers: None | Logger | Iterable = None,
                 ) -> None:
        """
        Initializes the trainer

        Args:
            img (np.ndarray): Target image
            model (nn.Module): Model to train
            optimizer_type (Optimizer, optional): Type of the optimizer.
                Should be defined in torch.optim. Defaults to Adam.
            lr (float, optional): learning_rate. Defaults to 1e-3.
            batch_size (int, optional): batch size. Defaults to 4096.
            dataloader_workers (int, optional): number of workers for
            dataloaders. Defaults to 4.
            device (Optional[str], optional): # The device's
                specification complying with pytorch convention.
                None means auto-choice. Defaults to None.
            loggers (None | Logger  |  Iterable, optional): A list of loggers.
                Defaults to None.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self._model = model.to(self.device)

        self._loss_fn = nn.MSELoss()
        self._optimizer = optimizer_type(self._model.parameters(), lr)

        self._metric_accum = MSEMetricsAccumulator()
        self._img_accum = ImgDataAccumulator(img.shape)

        self._train_loader, self.val_loader = self._init_dataloaders(
            img, batch_size, dataloader_workers
        )

        self._logging_data = LoggingData(model=model)
        if loggers is None:
            self._loggers = ()
        elif not isinstance(loggers, Iterable):
            self._loggers = (loggers, )
        else:
            self._loggers = loggers

    def _batch_step(self, batch: Tuple) -> Tuple:
        pos, target = batch
        pos = pos.to(self.device)
        target = target.to(self.device)
        pred = self._model(pos)
        loss = self._loss_fn(pred, target.float())

        pos = pos.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return loss, pos, pred

    def _train(self, pbar: tqdm) -> None:
        for batch in self._train_loader:
            self._model.train()
            self._optimizer.zero_grad()
            loss, pos, pred = self._batch_step(batch)
            loss.backward()

            self._metric_accum.update(
                loss.item(), num_new_samples=pos.shape[0])
            self._img_accum.update(pos, pred)
            pbar.update()
            pbar.set_postfix(
                train_mse=self._metric_accum.mse, train_psnr=self._metric_accum.psnr
            )

    def _val(self, pbar: tqdm) -> None:
        for batch in self.val_loader:
            self._model.eval()
            with torch.no_grad():
                loss, pos, pred = self._batch_step(batch)
                self._metric_accum.update(
                    loss.item(), num_new_samples=pos.shape[0]
                )
                self._img_accum.update(pos, pred)
                pbar.update()
                pbar.set_postfix(
                    val_mse=self._metric_accum.mse, val_psnr=self._metric_accum.psnr
                )

    def fit(self, num_epochs) -> None:
        """
        A method that runs the training with validation process

        Args:
            num_epochs (_type_): number of epochs
        """
        train_pbar = tqdm(total=len(self._train_loader), position=0)
        val_pbar = tqdm(total=len(self.val_loader), position=1)
        for epoch_id in range(num_epochs):
            self._logging_data.epoch_id = epoch_id
            self._metric_accum.reset()
            self._img_accum.reset()
            train_pbar.reset()
            val_pbar.reset()
            train_pbar.set_description_str(f'Train; Epoch {epoch_id}')
            val_pbar.set_description_str(f'Val; Epoch {epoch_id}')

            self._train(train_pbar)

            self._logging_data.train_mse = self._metric_accum.mse
            self._logging_data.train_psnr = self._metric_accum.psnr
            self._metric_accum.reset()

            self._optimizer.step()

            self._val(val_pbar)

            self._logging_data.val_mse = self._metric_accum.mse
            self._logging_data.val_psnr = self._metric_accum.psnr
            self._logging_data.pred_img = self._img_accum.img
            for logger in self._loggers:
                logger.log(self._logging_data)

            train_pbar.refresh()
            val_pbar.refresh()
        train_pbar.close()
        val_pbar.close()

    @staticmethod
    def _init_dataloaders(
        img: np.ndarray,
        batch_size: int,
        dataloader_workers: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Initializes dataloaders for training. Currently the next
        scheme is supported:training data includes positioins according
        to the slice [::2, ::2].  All other positions are used for the
        validation. The scheme results into the split: 0.25 pixels
        are used for training and 0.75 pixels -- for validation

        TODO: support other masks and make the method public.

        Args:
            img (np.ndarray): target image
            batch_size (int): batch size
            dataloader_workers (int): number of workers for
            dataloaders
        Returns:
            Tuple: _description_
        """
        train_mask = np.zeros(img.shape[:-1], dtype=bool)
        train_mask[::2, ::2] = True
        val_mask = np.logical_not(train_mask)

        train_dataloader = DataLoader(
            ImgDataset(img, mask=train_mask),
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
        )
        val_dataloader = DataLoader(
            ImgDataset(img, mask=val_mask),
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
        )
        return train_dataloader, val_dataloader
