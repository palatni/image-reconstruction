"""
This module contains pytorch models for training
"""
from typing import Iterable
from torch import nn
import torch


class FurierEncoder(nn.Module):
    """
    A Fourier encoder which. Its structure is described in the article [1]

    [1] Tancik, Matthew, et al. "Fourier features let networks learn high
    frequency functions in low dimensional domains." Advances in neural
    information processing systems 33 (2020): 7537-7547.
    """

    def __init__(
        self,
        num_phases: int,
        init_scale: float,
        trainable: bool = False,
    ) -> None:
        """
        Initializes the module.

        Args:
            num_phases (int): The number of (cos, sin) pairs in the
                encoder's output
            init_scale (float): The standard deviation of the normal
                distribution that initializes the weights.
            trainable (bool, optional): A flag indicating whether
                the encoder's parameters are trainable.
                According to the article [See class description]
                Training of the parameters is not affecting much
                the result. Defaults to False.
        """
        super().__init__()
        self._projector = nn.Parameter(
            data=torch.empty((2, num_phases)),
            requires_grad=trainable,
        )
        self._coef = nn.Parameter(
            data=torch.ones((1, num_phases * 2)),
            requires_grad=trainable,
        )
        torch.nn.init.normal_(tensor=self._projector, mean=0, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The model's forward pass method.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = x @ self._projector
        x = torch.view_as_real(torch.exp(2j * torch.pi * x))
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x * self._coef
        return x


class FurierMLP(nn.Module):
    def __init__(self,
                 num_phases: int,
                 encoder_scale: float,
                 mlp_feature_list: Iterable,
                 trainable_encoder: bool = False,
                 ) -> None:
        """
        A model consisting of encoder and MLP.

        Args:
            num_phases (int): The number of (cos, sin) pairs in the
                encoder's output
            encoder_scale (float): The standard deviation of the normal
                distribution that initializes the weights.
            mlp_feature_list (Sequence): a list consisting of output
                dimensions of each layer in the MLP.
            trainable (bool, optional): A flag indicating whether
                the encoder's parameters are trainable.
                According to the article [See class description]
                Training of the parameters is not affecting much
                the result. Defaults to False.
        """
        super().__init__()
        self._trainable = trainable_encoder

        self._encoder = FurierEncoder(
            num_phases=num_phases,
            init_scale=encoder_scale,
            trainable=trainable_encoder,
        )

        mlp_modules = []
        input_features = num_phases * 2
        for features in mlp_feature_list:
            mlp_modules.append(
                nn.Linear(in_features=input_features,
                          out_features=features, bias=True)
            )
            mlp_modules.append(nn.ReLU())
            input_features = features
        self._mlp = nn.Sequential(*mlp_modules)

        self._regressor = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The model's forward pass method.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self._encoder(x)
        x = self._mlp(x)
        x = self._regressor(x)
        return x
